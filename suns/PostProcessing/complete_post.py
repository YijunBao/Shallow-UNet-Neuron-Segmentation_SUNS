import io
import numpy as np
from scipy import sparse
from scipy import signal
from scipy.io import savemat, loadmat
import time
import multiprocessing as mp

from suns.PostProcessing.seperate_neurons import watershed_neurons, separate_neuron
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume
from suns.PostProcessing.refine_cons import refine_seperate, refine_seperate_multi


# %%
def complete_segment(pmaps: np.ndarray, Params: dict, useMP=True, useWT=False, display=False, p=None):
    '''Complete post-processing procedure. 
        This can be run after or before probablity thresholding, depending on whether Params['thresh_pmap'] is None.
        It first thresholds the "pmaps" (if Params['thresh_pmap'] is not None) into binary array, 
        then seperates the active pixels into connected regions, disgards regions smaller than Params['minArea'], 
        uses optional watershed (if useWT=True) to further segment regions larger than Params['avgArea'],
        merge the regions from different frames with close COM, large IoU, or large consume ratio,
        and finally selects masks that are active for at least Params['cons'] frames. 
        The output are "Masks_2", a 2D sparse matrix of the final segmented neurons,
        and "times_cons", a list of indices of frames when the final neuron is active.

    Inputs: 
        pmaps (3D numpy.ndarray of uint8, shape = (nframes,Lx,Ly)): the probability map obtained after CNN inference.
            If Params['thresh_pmap']==None, pmaps must be previously thresholded.
        Params (dict): Parameters for post-processing.
            Params['minArea']: Minimum area of a valid neuron mask (unit: pixels).
            Params['avgArea']: The typical neuron area (unit: pixels).
            Params['thresh_pmap']: The probablity threshold. Values higher than thresh_pmap are active pixels. 
                if Params['thresh_pmap']==None, then thresholding is not performed. 
                This is used when thresholding is done before this function.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params['thresh_COM']: Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['cons']: Minimum number of consecutive frames that a neuron should be active for.
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        useWT (bool, default to False): Indicator of whether watershed is used. 
        display (bool, default to False): Indicator of whether to show intermediate information
        p (multiprocessing.Pool, default to None): 

    Outputs:
        Masks_2 (sparse.csr_matrix of bool): the final segmented binary neuron masks after consecutive refinement. 
        times_cons (list of 1D numpy.array): indices of frames when the final neuron is active.
    '''
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    minArea = Params['minArea']
    avgArea = Params['avgArea']
    thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    start_all = time.time()

    # Segment neuron masks from each frame of probability map 
    start = time.time()
    if useMP:
        segs = p.starmap(separate_neuron, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1)
    else:
        segs =[separate_neuron(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps]
    end = time.time()
    num_neurons = sum([x[1].size for x in segs])
    if display:
        print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
            .format('separate Neurons', end-start,(end-start)/nframes*1000),
                '{:6d} segmented neurons.'.format(num_neurons))

    if num_neurons==0:
        print('No masks found. Please lower minArea or thresh_pmap.')
        Masks_2 = sparse.csc_matrix((0,Lx*Ly), dtype='bool')
        times_cons = []
    else: # find active neurons
        start = time.time()
        # Initally merge neurons with close COM.
        totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs)
        uniques, times_uniques = unique_neurons2_simp(totalmasks, neuronstate, COMs, \
            areas, probmapID, minArea=0, thresh_COM0=thresh_COM0)
        end_unique = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('unique_neurons1', end_unique - start, (end_unique - start) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_uniques)))

        # Further merge neurons with close COM.
        groupedneurons, times_groupedneurons = \
            group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
        end_COM = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('group_neurons', end_COM - end_unique, (end_COM - end_unique) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_groupedneurons)))

        # Merge neurons with high IoU.
        piecedneurons_1, times_piecedneurons_1 = \
            piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
        end_IOU = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('piece_neurons_IOU', end_IOU - end_COM, (end_IOU - end_COM) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_piecedneurons_1)))

        # Merge neurons with high consume ratio.
        piecedneurons, times_piecedneurons = \
            piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
        end_consume = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('piece_neurons_consume', end_consume - end_IOU, (end_consume - end_IOU) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_piecedneurons)))

        masks_final_2 = piecedneurons
        times_final = [np.unique(x) for x in times_piecedneurons]
            
        # Refine neurons using consecutive occurence requirement
        start = time.time()
        Masks_2, times_cons = refine_seperate(masks_final_2, times_final, cons, thresh_mask)
        end_all = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('refine_seperate', end_all - start, (end_all - start) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('Total time', end_all - start_all, (end_all - start_all) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))

    return Masks_2, times_cons


# %%
def optimize_combine_1(uniques: sparse.csr_matrix, times_uniques: list, dims: tuple, Params: dict, filename_GT: str):  
    '''Optimize 1 post-processing parameter: "cons". 
        Start after the first COM merging.
        The outputs are the recall, precision, and F1 calculated using all values in "list_cons".

    Inputs: 
        uniques (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the neuron masks to be merged.
        times_uniques (list of 1D numpy.array): indices of frames when the neuron is active.
        dims (tuple of int, shape = (2,)): the lateral shape of the image.
        Params (dict): Ranges of post-processing parameters to optimize over.
            Params['avgArea']: The typical neuron area (unit: pixels).
            Params['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params['thresh_COM']: (float or int) Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params['thresh_IOU']: (float) Threshold of IoU used for merging neurons.
            Params['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
            The GT masks are stored in a ".mat" file, and dataset "GTMasks_2" is the GT masks
            (shape = (Ly0*Lx0,n) when saved in MATLAB).

    Outputs:
        Recall_k (1D numpy.array of float): Recall for all cons. 
        Precision_k (1D numpy.array of float): Precision for all cons. 
        F1_k (1D numpy.array of float): F1 for all cons. 
    '''
    avgArea = Params['avgArea']
    thresh_mask = Params['thresh_mask']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = (1+thresh_IOU)/2
    list_cons = Params['list_cons']

    # second merge neurons with close COM.
    groupedneurons, times_groupedneurons = group_neurons(uniques, \
        thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques, useMP=False)
    # Merge neurons with high IoU.
    piecedneurons_1, times_piecedneurons_1 = piece_neurons_IOU(groupedneurons, \
        thresh_mask, thresh_IOU, times_groupedneurons)
    # Merge neurons with high consume ratio.
    piecedneurons, times_piecedneurons = piece_neurons_consume(piecedneurons_1, \
        avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
    masks_final_2 = piecedneurons
    times_final = [np.unique(x) for x in times_piecedneurons]

    data_GT=loadmat(filename_GT)
    GTMasks_2 = data_GT['GTMasks_2'].transpose()
    # Search for optimal "cons" used to refine segmented neurons.
    Recall_k, Precision_k, F1_k = refine_seperate_multi(GTMasks_2, \
        masks_final_2, times_final, list_cons, thresh_mask, display=False)
    return Recall_k, Precision_k, F1_k


def optimize_combine_3(totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set: dict, filename_GT: str, useMP=True, p=None):
    '''Optimize 3 post-processing parameters: "thresh_COM", "thresh_IOU", "cons". 
        Start before the first COM merging, which can include disgarding masks smaller than "minArea".
        The outputs are the recall, precisoin, and F1 calculated from all parameter combinations.

    Inputs: 
        totalmasks (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the neuron masks to be merged.
        neuronstate (1D numpy.array of bool, shape = (n,)): Indicators of whether a neuron is obtained without watershed.
        COMs (2D numpy.array of float, shape = (n,2)): COMs of the neurons.
        areas (1D numpy.array of uint32, shape = (n,)): Areas of the neurons. 
        probmapID (1D numpy.array of uint32, shape = (n,): indices of frames when the neuron is active. 
        dims (tuple of int, shape = (2,)): the lateral shape of the region.
        minArea (float or int, default to 0): Minimum area of a valid neuron mask (unit: pixels).
        avgArea (float or int, default to 0): The typical neuron area (unit: pixels). 
            Neuron masks with areas larger than avgArea will be further segmented by watershed.
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
            The GT masks are stored in a ".mat" file, and dataset "GTMasks_2" is the GT masks
            (shape = (Ly0*Lx0,n) when saved in MATLAB).
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 

    Outputs:
        list_Recall_inter (3D numpy.array of float): Recall for all paramter combinations. 
        list_Precision_inter (3D numpy.array of float): Precision for all paramter combinations. 
        list_F1_inter (3D numpy.array of float): F1 for all paramter combinations. 
            For these outputs, the orders of the tunable parameters are:
            "thresh_COM", "thresh_IOU", "cons"
    '''
    thresh_mask = Params_set['thresh_mask']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    list_cons = Params_set['list_cons']
    
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    size_inter = (L_thresh_COM, L_thresh_IOU, L_cons)

    # Initally merge neurons with close COM.
    uniques, times_uniques = unique_neurons2_simp(totalmasks, neuronstate, COMs, areas, probmapID, minArea, thresh_COM0, useMP=False)

    if not times_uniques:
        list_Recall_inter = np.zeros(size_inter)
        list_Precision_inter = np.zeros(size_inter)
        list_F1_inter = np.zeros(size_inter)
    else:
        # Calculate accuracy scores for various "thresh_COM", "thresh_IOU", and "cons".
        if useMP:
            # p3 = mp.Pool()
            list_temp = p.starmap(optimize_combine_1, [(uniques, times_uniques, dims, 
                {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                'thresh_IOU': thresh_IOU, 'list_cons':list_cons}, filename_GT) \
                    for thresh_COM in list_thresh_COM for thresh_IOU in list_thresh_IOU], chunksize=1)
            list_Recall_inter = np.vstack([x[0] for x in list_temp]).reshape(size_inter)
            list_Precision_inter = np.vstack([x[1] for x in list_temp]).reshape(size_inter)
            list_F1_inter = np.vstack([x[2] for x in list_temp]).reshape(size_inter)
            # p3.close()
            # p3.join()
        else: 
            list_Recall_inter = np.zeros(size_inter)
            list_Precision_inter = np.zeros(size_inter)
            list_F1_inter = np.zeros(size_inter)
            for (j1,thresh_COM) in enumerate(list_thresh_COM):
                for (j2,thresh_IOU) in enumerate(list_thresh_IOU):
                    (Recal_k, Precision_k, F1_k) = optimize_combine_1(uniques, times_uniques, dims, 
                        {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                            'thresh_IOU': thresh_IOU, 'list_cons':list_cons}, filename_GT)                                 
                    list_Recall_inter[j1,j2,:]=Recal_k
                    list_Precision_inter[j1,j2,:]=Precision_k
                    list_F1_inter[j1,j2,:]=F1_k

    return list_Recall_inter, list_Precision_inter, list_F1_inter

    
# %%
def parameter_optimization(pmaps: np.ndarray, Params_set: dict, \
        filename_GT: str, useMP=True, useWT=False, p=None): 
    '''Optimize 6 post-processing parameters over the entire post-processing procedure: 
        "minArea", "avgArea", "thresh_pmap", "thresh_COM", "thresh_IOU", "cons". 
        The outputs are the recall, precisoin, and F1 calculated from all parameter combinations.

    Inputs: 
        pmaps (3D numpy.ndarray of uint8, shape = (nframes,Lx,Ly)): the probability map obtained after CNN inference.
            It should not be be previously thresholded. if "thresh_pmap" is going to be optimized
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
            The GT masks are stored in a ".mat" file, and dataset "GTMasks_2" is the GT masks
            (shape = (Ly0*Lx0,n) when saved in MATLAB).
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        useWT (bool, default to False): Indicator of whether watershed is used. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        list_Recall (6D numpy.array of float): Recall for all paramter combinations. 
        list_Precision (6D numpy.array of float): Precision for all paramter combinations. 
        list_F1 (6D numpy.array of float): F1 for all paramter combinations. 
            For these outputs, the orders of the tunable parameters are:
            "minArea", "avgArea", "thresh_pmap", "thresh_COM", "thresh_IOU", "cons"
    '''
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    # thresh_mask = Params_set['thresh_mask']
    # thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    list_cons = Params_set['list_cons']

    L_minArea=len(list_minArea)
    L_avgArea=len(list_avgArea)
    L_thresh_pmap=len(list_thresh_pmap)
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    # Remember the order of the tunable parameters
    dim_result = (L_minArea, L_avgArea, L_thresh_pmap, L_thresh_COM, L_thresh_IOU, L_cons)
    # 6D arrays to store the recall, precision, and F1 for all parameter combinations
    list_Recall = np.zeros(dim_result)
    list_Precision = np.zeros(dim_result)
    list_F1 = np.zeros(dim_result)

    if useMP and not p: # start a multiprocessing.Pool
        p = mp.Pool(mp.cpu_count())
        closep = True
    else:
        closep = False
    start = time.time()

    for (i3,thresh_pmap) in enumerate(list_thresh_pmap):
        print('Using thresh_pmap={}'.format(thresh_pmap))
        minArea = min(list_minArea)
        # Segment neuron masks from each frame of probability map 
        # without watershed using the smallest minArea
        if useMP:
            segs = p.starmap(separate_neuron, [(frame, thresh_pmap, minArea, 0, False) for frame in pmaps], chunksize=1)
        else:
            segs = [separate_neuron(frame, thresh_pmap, minArea, 0, False) for frame in pmaps]
        print('Used {} s'.format(time.time() - start))

        for (i2,avgArea) in enumerate(list_avgArea):
            if useWT: # Try to further segment large masks using watershed.
                print('Using avgArea={}, thresh_pmap={}'.format(avgArea, thresh_pmap))
                if useMP:
                    segs2 = p.starmap(watershed_neurons, [((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs], chunksize=32)
                else:
                    segs2 = [watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs]
                print('Used {} s'.format(time.time() - start))
            else: # for no watershed
                segs2 = segs

            totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs2)

            num_neurons = neuronstate.size
            if num_neurons==0 or totalmasks.nnz/pmaps.size>0.1:
                # If too many pixels are active, "thresh_pmap" is likely too small, so ignore such result
                list_Recall[:,i2,i3,:,:,:]=0
                list_Precision[:,i2,i3,:,:,:]=0
                list_F1[:,i2,i3,:,:,:]=0
            else:
                # Calculate accuracy scores for various "minArea", "thresh_COM", "thresh_IOU", and "cons".
                # "avgArea" and "thresh_pmap" are fixed at this step, because they are searched in outer loops.
                if useMP:
                    try: 
                        list_result = p.starmap(optimize_combine_3, [(totalmasks, neuronstate, COMs, areas, \
                            probmapID, dims, minArea, avgArea, Params_set, filename_GT, False) for minArea in list_minArea])
                        for i1 in range(L_minArea):
                            list_Recall[i1,i2,i3,:,:,:]=list_result[i1][0]
                            list_Precision[i1,i2,i3,:,:,:]=list_result[i1][1]
                            list_F1[i1,i2,i3,:,:,:]=list_result[i1][2]
                        print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[:,i2,i3,:,:,:].max()))
                    except OverflowError: 
                        list_Recall[:,i2,i3,:,:,:]=0
                        list_Precision[:,i2,i3,:,:,:]=0
                        list_F1[:,i2,i3,:,:,:]=0
                        print('OverflowError. Size of totalmasks is larger than 4 GB. thresh_pmap is likely too low.')
                    except MemoryError: 
                        list_Recall[:,i2,i3,:,:,:]=0
                        list_Precision[:,i2,i3,:,:,:]=0
                        list_F1[:,i2,i3,:,:,:]=0
                        print('MemoryError. Too much memory is needed. thresh_pmap is likely too low.')

                else:
                    for (i1,minArea) in enumerate(list_minArea):
                        print('Using minArea={}, avgArea={}, thresh_pmap={}'.format(minArea, avgArea, thresh_pmap))
                        list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_3(
                            totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set, filename_GT, useMP=False, p=p)
                        list_Recall[i1,i2,i3,:,:,:]=list_Recall_inter
                        list_Precision[i1,i2,i3,:,:,:]=list_Precision_inter
                        list_F1[i1,i2,i3,:,:,:]=list_F1_inter
                        print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[i1,i2,i3,:,:,:].max()))

    if useMP and closep:
        p.close()
        p.join()
    return list_Recall, list_Precision, list_F1

