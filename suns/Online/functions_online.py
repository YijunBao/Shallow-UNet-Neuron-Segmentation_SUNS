# %%
import os
import sys
import math
import numpy as np
import time
import h5py
import pyfftw
from scipy import sparse
from scipy import signal
from scipy import special
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat, loadmat
import multiprocessing as mp

from suns.Online.par_online import fastlog_2, fastmask_2, fastexp_2, \
    fastconv_2, fastnormf_2, fastnormback_2, fastthreshold_2, fastmediansubtract_2
from suns.PostProcessing.seperate_neurons import separate_neuron
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume
from suns.PostProcessing.refine_cons import refine_seperate


def spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2):
    '''Apply spatial homomorphic filtering to the input image.
        bb = exp(IFFT(mask2*FFT(log(bb+1)))).

    Inputs: 
        bb(2D numpy.ndarray of float32): array storing the raw image.
        bf(2D numpy.ndarray of complex64): array to store the complex spectrum for FFT.
        fft_object_b(pyfftw.FFTW): Object for forward FFT.
        fft_object_c(pyfftw.FFTW): Object for inverse FFT.
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        No output, but "bb" is changed to the spatially filtered image during the function
    '''
    fastlog_2(bb)
    fft_object_b()
    fastmask_2(bf, mask2)
    fft_object_c()
    fastexp_2(bb)


def preprocess_online(bb, dimspad, dimsnb, med_frame3, frame_SNR=None, past_frames = None, \
        mask2=None, bf=None, fft_object_b=None, fft_object_c=None, Poisson_filt=np.array([1]), \
        useSF=True, useTF=True, useSNR=True, med_subtract=False, update_baseline=False):
    '''Pre-process the registered image in "bb" into an SNR image "frame_SNR" 
        using known median and median-based std in "med_frame3".
        It includes spatial filter, temporal filter, and SNR normalization. 
        Each step is optional.

    Inputs: 
        bb(3D numpy.ndarray of float32): array storing the raw image.
        dimspad (tuplel of int, shape = (2,)): lateral dimension of the padded images to be multiples of 8.
        dimsnb (tuplel of int, shape = (2,)): lateral dimension of the padded images for numba calculation in pre-processing.
        frame_SNR (2D empty numpy.ndarray of float32): empty array to store the SNR image.
        med_frame3 (3D numpy.ndarray of float32): median and median-based standard deviation from initial frames.
        past_frames (3D numpy.ndarray of float32, shape=(Lt,Lx,Ly)): the images from the last "Lt" frames.
            Theese images are after spatial fitering but before temporal filtering.
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
        bf(3D numpy.ndarray of complex64, default to None): array to store the complex spectrum for FFT.
        fft_object_b(pyfftw.FFTW, default to None): Object for forward FFT.
        fft_object_c(pyfftw.FFTW, default to None): Object for inverse FFT.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.

    Outputs:
        frame_SNR (2D numpy.ndarray of float32, shape = (Lx,Ly)): the SNR image obtained after pre-processing. 
        frame_tf (2D numpy.ndarray of float32, shape = (Lx,Ly)): the image obtained after temporal filtering but before SNR normalization. 
            if update_baseline==False, then this is just 0, because it is not used later. 
    '''
    (rowspad, colspad) = dimspad
    (rowsnb, colsnb) = dimsnb
    
    if useSF: # Homomorphic spatial filtering
        spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2)

    if useTF: # Temporal filtering
        past_frames[-1] = bb[:rowsnb, :colsnb]
        fastconv_2(past_frames[:, :rowspad, :colspad], frame_SNR[:rowspad, :colspad], Poisson_filt)
    else:
        frame_SNR = bb[:rowsnb, :colsnb]

    if med_subtract and not useSF: # Subtract every frame with its median.
        temp = np.zeros(frame_SNR.shape[:2], dtype = 'float32')
        fastmediansubtract_2(frame_SNR[:, :rowspad, :colspad], temp[:, :rowspad], 2)

    if update_baseline: # keep the temporally filtered frame, 
        # used for updating median and median-based standard deviation
        frame_tf = frame_SNR.copy()
    else:
        frame_tf = 0

    # Median computation and normalization
    if useSNR:
        fastnormf_2(frame_SNR[:rowspad, :colspad], med_frame3[:, :rowspad, :colspad])
    else:
        fastnormback_2(frame_SNR[:rowspad, :colspad], max(1, med_frame3[0,:colspad,:colspad].mean()))

    return frame_SNR, frame_tf


def CNN_online(frame_SNR, fff, dims=None):
    '''Use CNN model "fff" to infer the probability of each pixel being active for image "frame_SNR".

    Inputs: 
        frame_SNR (2D empty numpy.ndarray of float32): SNR image.
        fff(tf.keras.Model): CNN model.
        dims (tuplel of int, shape = (2,), default to None): lateral dimension of the output image.

    Outputs:
        frame_prob (2D empty numpy.ndarray of float32): probability map.
    '''
    if dims is None:
        dims = frame_SNR.shape
    frame_SNR = frame_SNR[np.newaxis,:,:,np.newaxis] 
    frame_prob = fff.predict(frame_SNR, batch_size=1)
    frame_prob = frame_prob.squeeze()[:dims[0], :dims[1]] 
    return frame_prob


def separate_neuron_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT=False):
    '''Segment the probability map "frame_prob" into connected regions representing neurons.
        It disgards the regions whose areas are smaller than "minArea".
        When useWT=True, it further tries to segment neurons whose areas are larger than "avgArea" using watershed. 
        The outputs are the segmented masks and some statistics (areas, centers, and whether they are from watershed)

    Inputs: 
        frame_prob (2D empty numpy.ndarray of float32): probability map.
        pmaps_b (2D empty numpy.ndarray of uint8): probability map.
        thresh_pmap_float(float, range in 0 to 1): Threshold of probablity map.
        minArea: Minimum area of a valid neuron mask (unit: pixels).
        avgArea: The typical neuron area (unit: pixels).
        useWT (bool, default to False): Indicator of whether watershed is used. 

    Outputs:
        segs (list): A list of segmented masks for every frame with statistics.
    '''
    # threshold the probability map to binary activity
    fastthreshold_2(frame_prob, pmaps_b, thresh_pmap_float)
    # spatial clustering each frame to form connected regions representing active neurons
    segs = separate_neuron(pmaps_b, None, minArea, avgArea, useWT)
    return segs


def refine_seperate_cons_online(times_temp, cons=1, have_cons=None):
    '''Select the segmented masks that satisfy consecutive frame requirement.
        The output is the indicators of whether each neuron satisfies the requirement.

    Inputs: 
        times_temp (list of 1D numpy.ndarray of int): indices of frames when each neuron is active.
        cons (int, default to 1): Minimum number of consecutive frames that a neuron should be active for.
        have_cons (1D numpy.ndarray of bool, default to None): indices of 
            whether each neuron satisfy consecutive frame requirement after the previous update.

    Outputs:
        have_cons (1D numpy.ndarray of bool): indices of 
            whether each neuron satisfy consecutive frame requirement after current update.
    '''
    if cons>1:
        if have_cons is None: # Initialize have_cons with a zeros array
            num_masks=len(times_temp)
            have_cons=np.zeros(num_masks, dtype='bool')
        for kk in np.logical_not(have_cons).nonzero()[0]:
            times_diff1 = times_temp[kk][cons-1:] - times_temp[kk][:1-cons]
            have_cons[kk] = np.any(times_diff1==cons-1)
    else: # If cons==1, then all masks satisfy consecutive frame requirement
        num_masks=len(times_temp)
        have_cons=np.ones(num_masks, dtype='bool')

    return have_cons


def select_cons(tuple_final):
    '''Select the segmented masks that satisfy consecutive frame requirement.
        The output is the binary masks of the neurons that satisfy the requirement.

    Inputs: 
        tuple_final (tuple, shape = (5,)):  Segmented masks with statistics.

    Outputs:
        Masksb_final(sparse.csc_matrix of bool): 2D representation of the segmented binary masks.
    '''
    Masksb_final, _, _, _, have_cons = tuple_final
    if np.any(have_cons):
        Masksb_final = [el for (bl,el) in zip(have_cons,Masksb_final) if bl]
    else:
        Masksb_final = sparse.csc_matrix((0,Masksb_final[0].shape[1]), dtype='bool')
    return Masksb_final


def merge_complete(segs, dims, Params):
    '''Temporally merge segmented masks in a few frames.
        The output are the merged neuron masks and their statistics 
        (acitve frame indices, areas, whether satisfy consecutive activation).

    Inputs: 
        segs (list): A list of segmented masks for every frame with statistics.
        dims (tuple of int, shape = (2,)): lateral dimension of the image.
        Params_post (dict): Parameters for post-processing.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params['thresh_COM']: Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['cons']: Minimum number of consecutive frames that a neuron should be active for.

    Outputs:
        Masks_2 (list of sparse.csr_matrix of bool, shape = (1,Lx*Ly)): 
            2D representation of each segmented binary mask.
        masks_final_2 (list of sparse.csr_matrix of float32, shape = (1,Lx*Ly)): 
            2D representation of each segmented real-number mask.
        times_final (list of 1D numpy.ndarray of int): 
            indices of frames when each neuron is active.
        area (1D numpy.ndarray of float32): areas of each mask.
        have_cons (1D numpy.ndarray of bool): 
            indices of whether each neuron satisfy consecutive frame requirement.
        The above outputs are often grouped into a tuple (shape = (5,)): 
            Segmented masks with statistics after update.
    '''
    avgArea = Params['avgArea']
    thresh_mask = Params['thresh_mask']
    thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']

    totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs)
    # Initally merge neurons with close COM.
    uniques, times_uniques = unique_neurons2_simp(totalmasks, neuronstate, COMs, \
        areas, probmapID, minArea=0, thresh_COM0=thresh_COM0)
    if uniques.size:
        # Further merge neurons with close COM.
        groupedneurons, times_groupedneurons = \
            group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques)
        # Merge neurons with high IoU.
        piecedneurons_1, times_piecedneurons_1 = \
            piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
        # Merge neurons with high consume ratio.
        piecedneurons, times_piecedneurons = \
            piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
        # %% Final result
        masks_final_2 = piecedneurons
        times_final = [np.unique(x) for x in times_piecedneurons]
            
        # %% Refine neurons using consecutive occurence
        if masks_final_2.size:
            masks_final_2 = [x for x in masks_final_2]
            Masks_2 = [(x >= x.max() * thresh_mask).astype('float') for x in masks_final_2]
            area = np.array([x.nnz for x in Masks_2])
            have_cons = refine_seperate_cons_online(times_final, cons)
        else:
            Masks_2 = []
            area = np.array([])
            have_cons = np.array([])

    else:
        Masks_2 = []
        masks_final_2 = []
        times_final = times_uniques
        area = np.array([])
        have_cons = np.array([])

    return Masks_2, masks_final_2, times_final, area, have_cons 


def merge_2(tuple1, tuple2, dims, Params):
    '''Merge newly segmented masks to previously segmented masks, together with their statistics
        (acitve frame indices, areas, whether satisfy consecutive activation).

    Inputs: 
        tuple1 (tuple, shape = (5,)): Segmented masks with statistics for the previous frames.
        tuple2 (tuple, shape = (5,)): Segmented masks with statistics for the new frames.
            tuple1 and tuple2 have the save format of the output tuple
        dims (tuple of int, shape = (2,)): lateral dimension of the image.
        Params_post (dict): Parameters for post-processing.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['cons']: Minimum number of consecutive frames that a neuron should be active for.

    Outputs:
        Masksb_merge (list of sparse.csr_matrix of bool, shape = (1,Lx*Ly)): 
            2D representation of each segmented binary mask.
        masks_merge (list of sparse.csr_matrix of float32, shape = (1,Lx*Ly)): 
            2D representation of each segmented real-number mask.
        times_merge (list of 1D numpy.ndarray of int): 
            indices of frames when each neuron is active.
        area_merge (1D numpy.ndarray of float32): areas of each mask.
        have_cons_merge (1D numpy.ndarray of bool): 
            indices of whether each neuron satisfy consecutive frame requirement.
        The above outputs are often grouped into a tuple (shape = (5,)): 
            Segmented masks with statistics after update.
    '''
    thresh_mask = Params['thresh_mask']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    
    Masksb2, masks2, times2, area2, have_cons2 = tuple2
    N2 = len(masks2)
    if N2==0: # If no additional masks is found in the new frames, the output is tuple1
        return tuple1
    if area2.ndim==0:
        area2 = np.expand_dims(area2, axis=0)
    
    Masksb1, masks1, times1, area1, have_cons1 = tuple1
    N1 = len(masks1)

    # Calculate IoU and consume ratio. Each row is an old mask, and each column is a new mask
    area1_2d = np.expand_dims(area1, axis=1).repeat(N2, axis=1)
    area2_2d = np.expand_dims(area2, axis=0).repeat(N1, axis=0)
    area_i = sparse.vstack(Masksb1).dot(sparse.vstack(Masksb2).T).toarray()
    area_u = area1_2d + area2_2d - area_i
    IOU = area_i/area_u
    consume = area_i/np.minimum(area1_2d, area2_2d)
    # pairs of neuron to be merged
    merge = np.logical_or.reduce([IOU>=thresh_IOU, consume>=thresh_consume])

    if not np.any(merge): # If no merge required, simply add tuple1 and tuple2
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge)

    # if a mask in masks2 is overlapped with multiple masks in makes1, keep only the one with the largest IOU
    merged = merge.sum(axis=0)
    multi = merged>1
    if np.any(multi): 
        for y in multi.nonzero()[0]:
            xs = merge[:,y].nonzero()[0]
            x0 = xs[IOU[xs,y].argmax()]
            for x in xs:
                if x!=x0:
                    merge[x,y] = 0

    # if a mask in masks1 is overlapped with multiple masks in makes2, merge the multiple masks in makes2 first
    mergedx = merge.sum(axis=1)
    multix = mergedx>1
    if np.any(multix): 
        for x in multix.nonzero()[0]:
            ys = merge[x,:].nonzero()[0]
            y0 = ys[0]
            mask2_merge = sum([masks2[ind] for ind in ys])
            masks2[y0] = mask2_merge
            times2[y0] = np.unique(np.hstack([times2[yi] for yi in ys]))
            merge[x,ys[1:]] = 0

    # finally merge masks1 and masks2
    (x, y) = merge.nonzero()
    for (xi, yi) in zip(x, y):
        mask_update = masks1[xi] + masks2[yi]
        Maskb_update = mask_update >= mask_update.max() * thresh_mask
        masks1[xi] = mask_update
        Masksb1[xi] = Maskb_update
        times1[xi] = np.hstack([times1[xi], times2[yi]])
        area1[xi] = Maskb_update.nnz
        if have_cons2[yi]:
            have_cons1[xi]=True
    have_cons1 = refine_seperate_cons_online(times1, cons, have_cons1)

    if np.all(merged): # If all new masks are merged to old masks, the mask list does not change
        return (Masksb1, masks1, times1, area1, have_cons1)
    else: # If some new masks are not merged to old masks, they should be added to the mask list
        unmerged = np.logical_not(merged).nonzero()[0]
        masks2 = [masks2[ind] for ind in unmerged]
        Masksb2 = [Masksb2[ind] for ind in unmerged]
        area2 = area2[unmerged]
        have_cons2 = have_cons2[unmerged]
        times2 = [times2[ind] for ind in unmerged]
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])

        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) 


def merge_2_nocons(tuple1, tuple2, dims, Params):
    '''Merge newly segmented masks to previously segmented masks
        that do not satisfy consecutive frame requirement.
        The output are the merged neuron masks and their statistics 
        (acitve frame indices, areas, whether satisfy consecutive activation).

    Inputs: 
        tuple1 (tuple, shape = (5,)): Segmented masks with statistics for the previous frames.
        tuple2 (tuple, shape = (5,)): Segmented masks with statistics for the new frames.
            tuple1 and tuple2 have the save format of the output tuple
        dims (tuple of int, shape = (2,)): lateral dimension of the image.
        Params (dict): Parameters for post-processing.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['cons']: Minimum number of consecutive frames that a neuron should be active for.

    Outputs:
        Masksb_merge (list of sparse.csr_matrix of bool, shape = (1,Lx*Ly)): 
            2D representation of each segmented binary mask.
        masks_merge (list of sparse.csr_matrix of float32, shape = (1,Lx*Ly)): 
            2D representation of each segmented real-number mask.
        times_merge (list of 1D numpy.ndarray of int): 
            indices of frames when each neuron is active.
        area_merge (1D numpy.ndarray of float32): areas of each mask.
        have_cons_merge (1D numpy.ndarray of bool): 
            indices of whether each neuron satisfy consecutive frame requirement.
        The above outputs are often grouped into a tuple (shape = (5,)): 
            Segmented masks with statistics after update.
    '''
    thresh_mask = Params['thresh_mask']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']

    Masksb2, masks2, times2, area2, have_cons2 = tuple2
    N2 = len(masks2)
    if N2==0: # If no additional masks is found in the new frames, the output is tuple1
        return tuple1
    if area2.ndim==0:
        area2 = np.expand_dims(area2, axis=0)

    Masksb1, masks1, times1, area1, have_cons1 = tuple1
    # Select neurons in tuple1 that do not satisfy consecutive frame requirement
    ind_nocons = np.logical_not(have_cons1).nonzero()[0]
    N1_nocons = ind_nocons.size
    if not N1_nocons: # If all neurons in tuple1 satisfy consecutive frame requirement,
        # then no merging can occur. Add tuple1 and tuple2 directly
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) 

    Masksb1_nocons = [Masksb1[ind] for ind in ind_nocons]
    area1_nocons = area1[ind_nocons]

    # Calculate IoU and consume ratio. Each row is an old mask, and each column is a new mask
    area1_2d = np.expand_dims(area1_nocons, axis=1).repeat(N2, axis=1)
    area2_2d = np.expand_dims(area2, axis=0).repeat(N1_nocons, axis=0)
    area_i = sparse.vstack(Masksb1_nocons).dot(sparse.vstack(Masksb2).T).toarray()
    area_u = area1_2d + area2_2d - area_i
    IOU = area_i/area_u
    consume_1 = area_i/area1_2d
    consume_2 = area_i/area2_2d
    # pairs of neuron to be merged
    merge = np.logical_or.reduce([IOU>=thresh_IOU, consume_1>=thresh_consume, consume_2>=thresh_consume])

    if not np.any(merge): # If no merge required, simply add tuple1 and tuple2
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge)

    # if a mask in masks2 is overlapped with multiple masks in makes1, keep only the one with the largest IOU
    merged = merge.sum(axis=0)
    multi = merged>1
    if np.any(multi): 
        for y in multi.nonzero()[0]:
            xs = merge[:,y].nonzero()[0]
            x0 = xs[IOU[xs,y].argmax()]
            for x in xs:
                if x!=x0:
                    merge[x,y] = 0

    # if a mask in masks1 is overlapped with multiple masks in makes2, merge the multiple masks in makes2 first
    mergedx = merge.sum(axis=1)
    multix = mergedx>1
    if np.any(multix): 
        for x in multix.nonzero()[0]:
            ys = merge[x,:].nonzero()[0]
            y0 = ys[0]
            mask2_merge = sum([masks2[ind] for ind in ys])
            masks2[y0] = mask2_merge
            times2[y0] = np.unique(np.hstack([times2[yi] for yi in ys]))
            merge[x,ys[1:]] = 0

    # finally merge masks1 and masks2
    (x, y) = merge.nonzero()
    for (xi, yi) in zip(x, y):
        xi0 = ind_nocons[xi] # xi is the index in the "nocons" masks. xi0 is the index in the full tuple1.
        mask_update = masks1[xi0] + masks2[yi]
        Maskb_update = mask_update >= mask_update.max() * thresh_mask
        masks1[xi0] = mask_update
        Masksb1[xi0] = Maskb_update
        times1[xi0] = np.hstack([times1[xi0], times2[yi]])
        area1[xi0] = Maskb_update.nnz
        if have_cons2[yi]:
            have_cons1[xi0]=True
    have_cons1 = refine_seperate_cons_online(times1, cons, have_cons1)

    if np.all(merged): # If all new masks are merged to old masks, the mask list does not change
        return (Masksb1, masks1, times1, area1, have_cons1)
    else: # If some new masks are not merged to old masks, they should be added to the mask list
        unmerged = np.logical_not(merged).nonzero()[0]
        masks2 = [masks2[ind] for ind in unmerged]
        Masksb2 = [Masksb2[ind] for ind in unmerged]
        area2 = area2[unmerged]
        have_cons2 = have_cons2[unmerged]
        times2 = [times2[ind] for ind in unmerged]
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])

        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge)


def final_merge(tuple_temp, Params):
    '''An extra round of merging at the end of online processing,
        to merge all previously detected neurons according their to IoU and consume ratio, like in batch mode.
        The output are "Masks_2b", a 2D sparse matrix of the final segmented neurons,
        and "times_cons", a list of indices of frames when the final neuron is active.

    Inputs: 
        tuple_temp (tuple, shape = (5,)): Segmented masks with statistics.
        Params (dict): Parameters for post-processing.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['cons']: Minimum number of consecutive frames that a neuron should be active for.
            Params['avgArea']: The typical neuron area (unit: pixels).

    Outputs:
        Masks_2b (sparse.csr_matrix of bool): the final segmented binary neuron masks after consecutive refinement. 
        times_cons (list of 1D numpy.array): indices of frames when the final neuron is active.
    '''
    _, masks, times, _, _ = tuple_temp
    if len(masks)==0: # If no masks is found, the output is tuple_temp
        return tuple_temp
    # if area.ndim==0:
    #     area = np.expand_dims(area, axis=0)
    thresh_mask = Params['thresh_mask']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    avgArea = Params['avgArea']
    masks = sparse.vstack(masks)

    masks_1, times_1 = piece_neurons_IOU(masks, thresh_mask, thresh_IOU, times)
    masks_final_2, times_2 = piece_neurons_consume(masks_1, avgArea, thresh_mask, thresh_consume, times_1)
    Masks_2b, times_final = refine_seperate(masks_final_2, times_2, cons, thresh_mask)

    return Masks_2b, times_final
