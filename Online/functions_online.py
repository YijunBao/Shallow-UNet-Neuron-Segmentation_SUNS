# %%
import os
import math
import numpy as np
import time
import h5py
import sys
import pyfftw
from scipy import sparse
from scipy import signal
from scipy import special
from scipy.optimize import linear_sum_assignment

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import par_online
import functions_init
import preprocessing_functions
from seperate_multi import separateNeuron
from combine import segs_results, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume


def spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2):
    '''Apply spatial filtering to the input image.

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
    par_online.fastlog(bb)
    fft_object_b()
    par_online.fastmask(bf, mask2)
    fft_object_c()
    par_online.fastexp(bb)


def preprocess_online(bb, dimspad, med_frame3, frame_SNR=None, past_frames = None, \
        mask2=None, bf=None, fft_object_b=None, fft_object_c=None, Poisson_filt=np.array([1]), \
        useSF=True, useTF=True, useSNR=True):
    '''Pre-process the registered image into an SNR image using known median and median-based std.

    Inputs: 
        bb(3D numpy.ndarray of float32): array storing the raw image.
        dimspad (tuplel of int, shape = (2,)): lateral dimension of the padded images.
        frame_SNR (2D empty numpy.ndarray of float32): empty array to store the SNR image.
        pmaps_b(3D empty numpy.ndarray of uint8): array to store the probablity map of the inital video.
        med_frame3 (3D empty numpy.ndarray of float32): median and median-based standard deviation from initial frames.
        recent_frames (3D numpy.ndarray of float32, shape=(Lt,Lx,Ly)): the images from the last "Lt" frames.
            Theese images are after spatial fitering but before temporal filtering.
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
        bf(3D numpy.ndarray of complex64, default to None): array to store the complex spectrum for FFT.
        fft_object_b(pyfftw.FFTW, default to None): Object for forward FFT.
        fft_object_c(pyfftw.FFTW, default to None): Object for inverse FFT.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.

    Outputs:
        frame_SNR (2D numpy.ndarray of float32, shape = (Lx,Ly)): the SNR image obtained after pre-processing. 
    '''
    (rowspad, colspad) = dimspad
    
    # %% Homomorphic spatial filtering
    if useSF:
        spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2)
        # par_online.fastlog(bb)
        # fft_object_b()
        # par_online.fastmask(bf, mask2)
        # fft_object_c()
        # par_online.fastexp(bb)

    # %% Temporal filtering
    if useTF:
        past_frames[-1] = bb[:rowspad, :colspad]
        par_online.fastconv(past_frames, frame_SNR, Poisson_filt)
    else:
        frame_SNR = bb[:rowspad, :colspad]

    # %% Median computation and normalization
    if useSNR:
        par_online.fastnormf(frame_SNR, med_frame3)
    else:
        par_online.fastnormback(frame_SNR, med_frame3[0,:,:].mean())

    return frame_SNR


def CNN_online(frame_SNR, fff, dims=None):
    '''Use CNN to inference the probability of each pixel being active for each image.

    Inputs: 
        frame_SNR (2D empty numpy.ndarray of float32): SNR image.
        fff(tf.keras.Model): CNN model.
        dims (tuplel of int, shape = (2,), default to None): lateral dimension of the image.

    Outputs:
        frame_prob (2D empty numpy.ndarray of float32): probability map.
    '''
    if dims is None:
        dims = frame_SNR.shape
    frame_SNR = frame_SNR[np.newaxis,:,:,np.newaxis] 
    frame_prob = fff.predict(frame_SNR, batch_size=1)
    frame_prob = frame_prob.squeeze()[:dims[0], :dims[1]] 
    return frame_prob


def postprocess_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT=False):
    '''Post-process the probability map.

    Inputs: 
        frame_prob (2D empty numpy.ndarray of float32): probability map.
        pmaps_b (2D empty numpy.ndarray of uint8): probability map.
        thresh_pmap_float(float, range in 0 to 1): Threshold of probablity map.
        minArea: Minimum area of a valid neuron mask (unit: pixels).
        avgArea: The typical neuron area (unit: pixels).
        useWT (bool, default to False): Indicator of whether watershed is used. 

    Outputs:
        segs (list): A list of outputs of "separateNeuron".
    '''
    # threshold the probability map to binary activity
    par_online.fastthreshold(frame_prob, pmaps_b, thresh_pmap_float)
    # spatial clustering each frame to form connected regions representing active neurons
    segs = separateNeuron(pmaps_b, None, minArea, avgArea, useWT)
    return segs


# def complete_segment_online(pmaps, Params, dims, frames_init, merge_every, display=True, track=False, useWT=False, p=None):
#     '''Complete online post-processing procedure. This can be run before or after probablity thresholding,
#         depending on whether Params['thresh_pmap'] is None.

#     Inputs: 
#         pmaps (3D numpy.ndarray of uint8, shape = (nframes,Lx,Ly)): the probability map obtained after CNN inference.
#             If Params['thresh_pmap']==None, pmaps must be previously thresholded.
#         Params (dict): Parameters for post-processing.
#             Params['minArea']: Minimum area of a valid neuron mask (unit: pixels).
#             Params['avgArea']: The typical neuron area (unit: pixels).
#             Params['thresh_pmap']: The probablity threshold. Values higher than thresh_pmap are active pixels. 
#                 if Params['thresh_pmap']==None, then thresholding is not performed. 
#                 This is used when thresholding is done before this function.
#             Params['thresh_mask']: Threashold to binarize the real-number mask.
#             Params['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
#             Params['thresh_COM']: Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
#             Params['thresh_IOU']: Threshold of IOU used for merging neurons.
#             Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
#             Params['cons']: Minimum number of consecutive frames that a neuron should be active for.
#         dims (tuple of int, shape = (2,)): the lateral shape of the region.
#         frames_init (intrue): indicator of whether multiprocessing is used to speed up. 
#         merge_every (int): Indicator of whether to show intermediate information
#         display (bool, default to False): Indicator of whether to show intermediate information
#         track (bool, default to False): Indicator of whether to show intermediate information
#         useWT (bool, default to False): Indicator of whether watershed is used. 
#         p (multiprocessing.Pool, default to None): 

#     Outputs:
#         Masks_2 (sparse.csr_matrix of bool): the final segmented binary neuron masks after consecutive refinement. 
#     '''
#     thresh_pmap = Params['thresh_pmap']
#     minArea = Params['minArea']
#     avgArea = Params['avgArea']
#     if p is None:
#         mp.Pool()
#     segs = p.starmap(separateNeuron, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
#     # Masks_2, _, _, _ = functions_online.merge_complete(segs, dims, Params)

#     tuple_init = merge_complete(segs[:frames_init], dims, Params)
#     if track:
#         Masks_2 = merge_final_track(tuple_init, segs[frames_init:], dims, Params, frames_init, merge_every) #, show_intermediate=True
#     else:
#         Masks_2 = merge_final(tuple_init, segs[frames_init:], dims, Params, frames_init, merge_every) #, show_intermediate=True
#     return Masks_2


def merge_complete(segs, dims, Params):
    # minArea = Params['minArea']
    avgArea = Params['avgArea']
    # thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']

    totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs)
    uniques, times_uniques = uniqueNeurons2_simp(totalmasks, neuronstate, COMs, \
        areas, probmapID, minArea=0, thresh_COM0=thresh_COM0)
    if uniques.size:
        groupedneurons, times_groupedneurons = \
            group_neurons(uniques, thresh_COM, thresh_mask, (dims[0], dims[1]), times_uniques)
        piecedneurons_1, times_piecedneurons_1 = \
            piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
        piecedneurons, times_piecedneurons = \
            piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
        # %% Final result
        masks_final_2 = piecedneurons
        times_final = [np.unique(x) for x in times_piecedneurons]
            
        # %% Refine neurons using consecutive occurence
        if masks_final_2.size:
            masks_final_2 = [x for x in masks_final_2]
            Masks_2 = [(x >= x.max() * thresh_mask).astype('float') for x in masks_final_2]
            area = np.array([x.nnz for x in Masks_2]) # Masks_2.sum(axis=1).A.squeeze()
            # if select_cons:
            have_cons = refine_seperate_cons(times_final, cons)
            # else:
                # have_cons=np.zeros(len(masks_final_2), dtype='bool')
        else:
            Masks_2 = []
            area = np.array([])
            have_cons = np.array([])

    else:
        Masks_2 = []
        masks_final_2 = [] # masks_2_float = 
        times_final = times_uniques
        area = np.array([])
        have_cons = np.array([])

    return Masks_2, masks_final_2, times_final, area, have_cons 


def merge_2(tuple1, tuple2, dims, Params):
    # minArea = Params['minArea']
    # avgArea = Params['avgArea']
    # thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    # thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    Masksb1, masks1, times1, area1, have_cons1 = tuple1
    Masksb2, masks2, times2, area2, have_cons2 = tuple2

    N2= len(masks2)
    if N2==0:
        return tuple1
    N1= len(masks1)
    if area2.ndim==0:
        area2 = np.expand_dims(area2, axis=0)

    area1_2d = np.expand_dims(area1, axis=1).repeat(N2, axis=1)
    area2_2d = np.expand_dims(area2, axis=0).repeat(N1, axis=0)
    # area_i = Masksb1.dot(Masksb2.T).toarray()
    area_i = sparse.vstack(Masksb1).dot(sparse.vstack(Masksb2).T).toarray()
    # area_i = np.array([[x.dot(y.T)[0,0] for y in Masksb2] for x in Masksb1])
    area_u = area1_2d + area2_2d - area_i
    IOU = area_i/area_u
    consume_1 = area_i/area1_2d
    consume_2 = area_i/area2_2d
    merge = np.logical_or.reduce([IOU>=thresh_IOU, consume_1>=thresh_consume, consume_2>=thresh_consume])

    if not np.any(merge):
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) # Masks_2, masks_2_float, 

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
    mergedx = merge.sum(axis=0)
    multix = mergedx>1
    if np.any(multix): 
        for x in multix.nonzero()[0]:
            ys = merge[x,:].nonzero()[0]
            y0 = ys[0]
            mask2_merge = sum([masks2[ind] for ind in ys])
            masks2[y0] = mask2_merge
            times2[y0] = np.unique(np.hstack([times2[yi] for yi in ys]))
            merge[x,ys[1:]] = 0
            # not necessary
            # Maskb2_merge = mask2_merge >= mask2_merge.max() * thresh_mask
            # Maskb2 = sparse.vstack([Maskb2[:y0], Maskb2_merge, Maskb2[(y0+1):]])
            # area2[y0] = Maskb2_merge.nnz

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
    have_cons1 = refine_seperate_cons(times1, cons, have_cons1)

    if np.all(merged):
        return (Masksb1, masks1, times1, area1, have_cons1)
    else:
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

        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) # Masks_2, masks_2_float, 


def merge_2_nocons(tuple1, tuple2, dims, Params): # only merge masks2 to masks1 that do not have cons
    # minArea = Params['minArea']
    # avgArea = Params['avgArea']
    # thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    # thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']

    Masksb2, masks2, times2, area2, have_cons2 = tuple2
    N2 = len(masks2)
    if not N2:
        return tuple1
    if area2.ndim==0:
        area2 = np.expand_dims(area2, axis=0)

    Masksb1, masks1, times1, area1, have_cons1 = tuple1
    ind_nocons = np.logical_not(have_cons1).nonzero()[0]
    N1_nocons = ind_nocons.size
    if not N1_nocons:
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) # Masks_2, masks_2_float, 

    # N1= len(masks1)
    Masksb1_nocons = [Masksb1[ind] for ind in ind_nocons]
    # masks1_nocons = [masks1[ind] for ind in ind_nocons]
    # times1_nocons = [times1[ind] for ind in ind_nocons]
    area1_nocons = area1[ind_nocons]

    area1_2d = np.expand_dims(area1_nocons, axis=1).repeat(N2, axis=1)
    area2_2d = np.expand_dims(area2, axis=0).repeat(N1_nocons, axis=0)
    # area_i = Masksb1_nocons.dot(Masksb2.T).toarray()
    area_i = sparse.vstack(Masksb1_nocons).dot(sparse.vstack(Masksb2).T).toarray()
    # area_i = np.array([[x.dot(y.T)[0,0] for y in Masksb2] for x in Masksb1_nocons])
    area_u = area1_2d + area2_2d - area_i
    IOU = area_i/area_u
    consume_1 = area_i/area1_2d
    consume_2 = area_i/area2_2d
    merge = np.logical_or.reduce([IOU>=thresh_IOU, consume_1>=thresh_consume, consume_2>=thresh_consume])

    if not np.any(merge):
        masks_merge = masks1 + masks2
        times_merge = times1 + times2
        Masksb_merge = Masksb1 + Masksb2
        area_merge = np.hstack([area1, area2])
        have_cons_merge = np.hstack([have_cons1, have_cons2])
        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) # Masks_2, masks_2_float, 

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
    mergedx = merge.sum(axis=0)
    multix = mergedx>1
    if np.any(multix): 
        for x in multix.nonzero()[0]:
            ys = merge[x,:].nonzero()[0]
            y0 = ys[0]
            mask2_merge = sum([masks2[ind] for ind in ys])
            masks2[y0] = mask2_merge
            times2[y0] = np.unique(np.hstack([times2[yi] for yi in ys]))
            merge[x,ys[1:]] = 0
            # not necessary
            # Maskb2_merge = mask2_merge >= mask2_merge.max() * thresh_mask
            # Maskb2 = sparse.vstack([Maskb2[:y0], Maskb2_merge, Maskb2[(y0+1):]])
            # area2[y0] = Maskb2_merge.nnz

    # finally merge masks1 and masks2
    (x, y) = merge.nonzero()
    for (xi, yi) in zip(x, y):
        xi0 = ind_nocons[xi]
        mask_update = masks1[xi0] + masks2[yi]
        Maskb_update = mask_update >= mask_update.max() * thresh_mask
        masks1[xi0] = mask_update
        Masksb1[xi0] = Maskb_update
        times1[xi0] = np.hstack([times1[xi0], times2[yi]])
        area1[xi0] = Maskb_update.nnz
        if have_cons2[yi]:
            have_cons1[xi0]=True
    have_cons1 = refine_seperate_cons(times1, cons, have_cons1)

    if np.all(merged):
        return (Masksb1, masks1, times1, area1, have_cons1)
    else:
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

        return (Masksb_merge, masks_merge, times_merge, area_merge, have_cons_merge) # Masks_2, masks_2_float, 


def merge_final_track(tuple_temp, segs, dims, Params, frames_initf, merge_every):
    # # minArea = Params['minArea']
    # avgArea = Params['avgArea']
    # # thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    # thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    # cons = Params['cons']
    nframes_online = len(segs)
    (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp) = tuple_temp
    Masks_cons = select_cons(tuple_temp)
    Masks_cons_2D = sparse.vstack(Masks_cons) 
    ind_cons = have_cons_temp.nonzero()[0]
    N1 = len(Masks_cons)
    (Lx, Ly) = dims
    list_segs_new = []
    list_masks_old = [[] for _ in range(N1)]
    times_active_old = [[] for _ in range(N1)]
    active_old_previous = np.zeros(N1, dtype='bool')
    segs0 = segs[0]
    segs_empty = (segs0[0][0:0], segs0[1][0:0], segs0[2][0:0], segs0[3][0:0])

    for t in range(0, nframes_online):
        if t%1000 ==0:
            print('{} frames have been processed online'.format(t))
        active_old = np.zeros(N1, dtype='bool')
        masks_t, neuronstate_t, cents_t, areas_t = segs[t]
        N2 = neuronstate_t.size
        if N2:
            new_found = np.zeros(N2, dtype='bool')
            for n2 in range(N2):
                masks_t2 = masks_t[n2]
                cents_t2 = np.round(cents_t[n2,1]) * Ly + np.round(cents_t[n2,0])  
                # possible_masks1 = Masks_cons[:,cents_t2].nonzero()[0]
                # possible_masks1 = [ind for ind in range(N1) if Masks_cons[ind][0,cents_t2]]
                possible_masks1 = Masks_cons_2D[:,cents_t2].nonzero()[0]
                # N1c = possible_masks1.size
                IOUs = np.zeros(len(possible_masks1))
                areas_t2 = areas_t[n2]
                for (ind,n1) in enumerate(possible_masks1):
                    # area_i = Masks_cons[n1].dot(masks_t2.T)[0,0]
                    # if not neuronstate_t[n2]:
                    #     area_i = area_i * 4
                    area_i = Masks_cons[n1].multiply(masks_t2).nnz
                    area_temp1 = area_temp[n1]
                    area_u = area_temp1 + areas_t2 - area_i
                    IOU = area_i / area_u
                    consume = area_i / min(area_temp1, areas_t2)
                    contain = (IOU >= thresh_IOU) or (consume >= thresh_consume)
                    if contain:
                        IOUs[ind] = IOU
                num_contains = IOUs.nonzero()[0].size
                if num_contains:
                    belongs = possible_masks1[IOUs.argmax()]
                    list_masks_old[belongs].append(masks_t2)
                    times_active_old[belongs].append(t + frames_initf)
                    active_old[belongs] = True
                else:
                    new_found[n2] = True

            if np.any(new_found):
                segs_new = (masks_t[new_found], neuronstate_t[new_found], cents_t[new_found], areas_t[new_found])
            else:
                segs_new = segs_empty # .copy()
                # segs_new = (masks_t[0:0], neuronstate_t[0:0], cents_t[0:0], areas_t[0:0])
                
        else:
            segs_new = segs[t]
        list_segs_new.append(segs_new)

        if ((t+1) % merge_every) == 0:
            # delay merging to next frame, to reserve merging time for new neurons
            active_old_previous = np.logical_and(active_old_previous, active_old)
        else:
            inactive = np.logical_and(active_old_previous, np.logical_not(active_old)).nonzero()[0]
            active_old_previous = active_old.copy()
            for n1 in inactive: # merge to already found neurons
                n10 = ind_cons[n1]
                mask_update = masks_temp[n10] + sum(list_masks_old[n1])
                masks_temp[n10] = mask_update
                times_add = np.unique(np.array(times_active_old[n1]))
                times_temp[n10] = np.hstack([times_temp[n10], times_add])
                list_masks_old[n1] = []
                times_active_old[n1] = []
                Maskb_update = mask_update >= mask_update.max() * thresh_mask
                Masksb_temp[n10] = Maskb_update
                Masks_cons[n1] = Maskb_update
                area_temp[n10] = Maskb_update.nnz
            if inactive.size:
                Masks_cons_2D = sparse.vstack(Masks_cons) 
                # masks_show = sum(Masksb_temp).toarray().reshape(Lx,Ly)
                # plt.imshow(masks_show>0); plt.show()


        if ((t+1) % merge_every) == 0: # merge newly found neurons 
            tuple_temp = (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp)
            tuple_add = merge_complete(list_segs_new, dims, Params)
            (Masksb_add, masks_add, times_add, area_add, have_cons_add) = tuple_add
            times_add = [x + t + frames_initf for x in times_add]
            tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)
            # tuple_temp = merge_2_Jaccard(tuple_temp, tuple_add, dims, Params)
            tuple_temp = merge_2_nocons(tuple_temp, tuple_add, dims, Params)

            (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp) = tuple_temp
            ind_cons_new = have_cons_temp.nonzero()[0]
            for (ind,ind_cons_0) in enumerate(ind_cons_new):
                if ind_cons_0 not in ind_cons:
                    if ind_cons_0 > ind_cons.max():
                        list_masks_old.append([])
                        times_active_old.append([])
                    else:
                        list_masks_old.insert(ind, [])
                        times_active_old.insert(ind, [])
            Masks_cons = select_cons(tuple_temp)
            Masks_cons_2D = sparse.vstack(Masks_cons) 
            N1 = len(Masks_cons)
            list_segs_new = []
            # list_masks_old = [[]] * N1
            # times_active_old = [[]] * N1
            active_old_previous = np.zeros_like(have_cons_temp) # np.zeros(N1, dtype='bool')
            active_old_previous[ind_cons] = active_old
            active_old_previous = active_old_previous[ind_cons_new]
            ind_cons = ind_cons_new

        if t == nframes_online-1: # last merge
            inactive = active_old_previous.nonzero()[0]
            for n1 in inactive: # merge to already found neurons
                n10 = ind_cons[n1]
                mask_update = masks_temp[n10] + sum(list_masks_old[n1])
                masks_temp[n10] = mask_update
                times_add = np.unique(np.array(times_active_old[n1]))
                times_temp[n10] = np.hstack([times_temp[n10], times_add])
                # list_masks_old[belongs] = []
                # times_active_old[belongs] = []
                Maskb_update = mask_update >= mask_update.max() * thresh_mask
                Masksb_temp[n10] = Maskb_update
                # Masks_cons[n1] = Maskb_update
                area_temp[n10] = Maskb_update.nnz
            # masks_show = sum(Masksb_temp).toarray().reshape(Lx,Ly)
            # plt.imshow(masks_show>0); plt.show()

            tuple_temp = (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp)
            if list_segs_new:
                tuple_add = merge_complete(list_segs_new, dims, Params)
                (Masksb_add, masks_add, times_add, area_add, have_cons_add) = tuple_add
                times_add = [x + t + frames_initf for x in times_add]
                tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)
                # tuple_temp = merge_2_Jaccard(tuple_temp, tuple_add, dims, Params)
                tuple_temp = merge_2_nocons(tuple_temp, tuple_add, dims, Params)
                # (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp) = tuple_temp

            Masks_cons = select_cons(tuple_temp)

    if len(Masks_cons):
        Masks_2 = sparse.vstack(Masks_cons)
    else:
        Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))
    return Masks_2 # , tuple_temp


def merge_final(tuple_temp, segs, dims, Params, frames_initf, merge_every, show_intermediate=True):
    nframes_online = len(segs)

    if show_intermediate:
        Masks_2 = select_cons(tuple_temp)

    for t in range(0, nframes_online, merge_every):
        if (t//merge_every)%30 ==0:
            print('{} frames have been processed'.format(t))
        tuple_add = merge_complete(segs[t:(t+merge_every)], dims, Params)
        (Masksb_add, masks_add, times_add, area_add, have_cons_add) = tuple_add
        times_add = [x + t + frames_initf for x in times_add]
        tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)
        # tuple_temp = merge_2_Jaccard(tuple_temp, tuple_add, dims, Params)
        tuple_temp = merge_2(tuple_temp, tuple_add, dims, Params)

        # masks_all = sparse.vstack([masks_temp, masks_add])
        # times_all = times_temp + [x + t + frames_initf for x in times_add]
        # masks_temp, times_temp = merge_2_slow(masks_all, times_all, dims, Params)

        # The following two lines are only used to show intermediate result
        if show_intermediate:
            Masks_2 = select_cons(tuple_temp)

    if not show_intermediate:
        Masks_2 = select_cons(tuple_temp)
    if len(Masks_2):
        Masks_2 = sparse.vstack(Masks_2)
    else:
        Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))
    return Masks_2 # , tuple_temp


def refine_seperate_cons(times_temp, cons=1, have_cons=None):
    if cons>1:
        if have_cons is None:
            num_masks=len(times_temp)
            have_cons=np.zeros(num_masks, dtype='bool')
        for kk in np.logical_not(have_cons).nonzero()[0]:
            times_diff1 = times_temp[kk][cons-1:] - times_temp[kk][:1-cons]
            have_cons[kk] = np.any(times_diff1==cons-1)
        # if np.any(have_cons):
        #     Masksb_temp = Masksb_temp[have_cons]
        # else:
        #     Masksb_temp = sparse.csc_matrix((0,Masksb_temp.shape[1]), dtype='bool')
    else:
        num_masks=len(times_temp)
        have_cons=np.ones(num_masks, dtype='bool')

    return have_cons


def select_cons(tuple_final):
    Masksb_final, _, _, _, have_cons = tuple_final
    if np.any(have_cons):
        Masksb_final = [el for (bl,el) in zip(have_cons,Masksb_final) if bl]
    else:
        Masksb_final = sparse.csc_matrix((0,Masksb_final[0].shape[1]), dtype='bool')
    return Masksb_final


def refine_seperate_cons_new(tuple_temp, cons=1):
    Masksb_temp, _, times_temp, _ = tuple_temp
    num_masks=len(times_temp)
    if cons>1:
        have_cons=np.zeros(num_masks, dtype='bool')
        for kk in range(num_masks):
            times_diff1 = times_temp[kk][cons-1:] - times_temp[kk][:1-cons]
            have_cons[kk] = np.any(times_diff1==cons-1)
        if np.any(have_cons):
            Masks_2 = Masksb_temp[have_cons]
        else:
            # print('No masks found. Please lower cons.')
            Masks_2 = sparse.csc_matrix((0,Masksb_temp.shape[1]), dtype='bool')
    else:
        Masks_2 = Masksb_temp

    return Masks_2


def online_processing_all(video_raw, dimspad, med_frame3, frames_init, merge_every, fff, Params_post, \
        segs_all, tuple_temp, mask2=None, cc2=None, \
        frame_SNR=None, past_frames = None, Poisson_filt=np.array([1]), pmaps_b=None, \
        useSF=True, useTF=True, useSNR=True, useWT=False, show_intermediate=True, display=False):
    # %% Online processing for the following frames
    leng_tf = Poisson_filt.size
    (nframes, Lx, Ly) = video_raw.shape
    dims = (Lx, Ly)
    nframesf = nframes - leng_tf + 1
    frames_initf = frames_init - leng_tf + 1
    current_frame = leng_tf
    leng_past = 2*leng_tf
    t_merge = frames_initf
    list_time_per=np.zeros(nframesf)

    minArea = Params_post['minArea']
    avgArea = Params_post['avgArea']
    # thresh_pmap = Params_post['thresh_pmap']
    thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    thresh_mask = Params_post['thresh_mask']
    thresh_COM0 = Params_post['thresh_COM0']
    thresh_COM = Params_post['thresh_COM']
    thresh_IOU = Params_post['thresh_IOU']
    thresh_consume = Params_post['thresh_consume']
    cons = Params_post['cons']

    # Spatial filtering preparation for online processing. 
    # Attention: this part counts to the total time
    if useSF:
        if cc2:
            pyfftw.import_wisdom(cc2)
        (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft2(dimspad)
    else:
        (bf, fft_object_b, fft_object_c) = (None, None, None)
        bb=np.zeros(dimspad, dtype='float32')
    
    for t in range(frames_initf,nframesf):
        if display:
            start_frame = time.time()
        bb[:Lx, :Ly] = video_raw[t]
        
        # PreProcessing
        frame_SNR = preprocess_online(bb, dimspad, med_frame3, frame_SNR, \
            past_frames[current_frame-leng_tf:current_frame], mask2, bf, fft_object_b, fft_object_c, \
            Poisson_filt, useSF=useSF, useTF=useTF, useSNR=useSNR)

        # if useSF: # Spatial filtering
        #     par_online.fastlog(bb)
        #     fft_object_b()
        #     par_online.fastmask(bf, mask2)
        #     fft_object_c()
        #     par_online.fastexp(bb)

        # if useTF: # Temporal filtering
        #     past_frames[current_frame] = bb[:rowspad, :colspad]
        #     par_online.fastconv(past_frames[current_frame-leng_tf:current_frame], frame_SNR, Poisson_filt)
        # else:
        #     frame_SNR = bb[:rowspad, :colspad]

        # if useSNR: # SNR normalization
        #     par_online.fastnormf(frame_SNR, med_frame3)
        # else:
        #     par_online.fastnormback(frame_SNR, 0, med_frame3[0,:,:].mean())

        # CNN inference
        frame_prob = CNN_online(frame_SNR, fff, dims)
        # frame_SNR_exp = frame_SNR[np.newaxis,:,:,np.newaxis]
        # frame_prob = fff.predict(frame_SNR_exp, batch_size=1)
        # frame_prob = frame_prob.squeeze()[:dims[0], :dims[1]]

        # post-processing
        segs = postprocess_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)
        # # threshold the probability map to binary activity
        # par_online.fastthreshold(frame_prob, pmaps_b, thresh_pmap_float)
        # # spatial clustering each frame to form connected regions representing active neurons
        # segs = separateNeuron(pmaps_b, None, minArea, avgArea, useWT)
        segs_all.append(segs)

        # temporal merging 1: combine neurons with COM distance smaller than thresh_COM0
        if ((t + 1 - t_merge) == merge_every) or (t==nframesf-1):
            # uniques, times_uniques = uniqueNeurons1_simp(segs_all[t_merge:(t+1)], thresh_COM0) # minArea,
            totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs_all[t_merge:(t+1)])
            uniques, times_uniques = uniqueNeurons2_simp(totalmasks, neuronstate, COMs, \
                areas, probmapID, minArea=0, thresh_COM0=thresh_COM0)

        # temporal merging 2: combine neurons with COM distance smaller than thresh_COM
        if ((t + 0 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                groupedneurons, times_groupedneurons = \
                    group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques)

        # temporal merging 3: combine neurons with IoU larger than thresh_IOU
        if ((t - 1 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                piecedneurons_1, times_piecedneurons_1 = \
                    piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)

        # temporal merging 4: combine neurons with conumse ratio larger than thresh_consume
        if ((t - 2 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                piecedneurons, times_piecedneurons = \
                    piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
                # masks of new neurons
                masks_add = piecedneurons
                # indices of frames when the neurons are active
                times_add = [np.unique(x) + t_merge for x in times_piecedneurons]
                    
                # Refine neurons using consecutive occurence
                if masks_add.size:
                    # new real-number masks
                    masks_add = [x for x in masks_add]
                    # new binary masks
                    Masksb_add = [(x >= x.max() * thresh_mask).astype('float') for x in masks_add]
                    # areas of new masks
                    area_add = np.array([x.nnz for x in Masksb_add])
                    # indicators of whether the new masks satisfy consecutive frame requirement
                    have_cons_add = refine_seperate_cons(times_add, cons)
                else:
                    Masksb_add = []
                    area_add = np.array([])
                    have_cons_add = np.array([])

            else: # does not find any active neuron
                Masksb_add = []
                masks_add = []
                times_add = times_uniques
                area_add = np.array([])
                have_cons_add = np.array([])
            tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)

        # temporal merging 5: merge newly found neurons within the recent "merge_every" frames with existing neurons
        if ((t - 3 - t_merge) == merge_every) or (t==nframesf-1):
            # tuple_temp = merge_2_Jaccard(tuple_temp, tuple_add, dims, Params)
            tuple_temp = merge_2(tuple_temp, tuple_add, dims, Params_post)
            t_merge = t+1
            if show_intermediate:
                Masks_2 = select_cons(tuple_temp)

        current_frame +=1
        # Update the stored latest frames when it runs out: move them "leng_tf" ahead
        if current_frame >= leng_past:
            current_frame = leng_tf
            past_frames[:leng_tf] = past_frames[-leng_tf:]
        if display:
            end_frame = time.time()
            list_time_per[t] = end_frame - start_frame
            if t % 1000 == 0:
                print('{} frames has been processed'.format(t))

    if not show_intermediate:
        Masks_2 = select_cons(tuple_temp)

    # final result. Masks_2 is a 2D sparse matrix of the segmented neurons
    if len(Masks_2):
        Masks_2 = sparse.vstack(Masks_2)
    else:
        Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))

    return Masks_2, list_time_per