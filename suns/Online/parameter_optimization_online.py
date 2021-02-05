import sys
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat
import time
import multiprocessing as mp

from suns.Online.functions_online import merge_2
from suns.PostProcessing.seperate_neurons import watershed_neurons, separate_neuron
from suns.PostProcessing.combine import unique_neurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume
from suns.PostProcessing.refine_cons import refine_seperate_multi


def merge_complete_nocons(uniques, times_uniques, dims, Params):
    '''Temporally merge segmented masks in a few frames. Used for parameter optimization.
        Ignore consecutive frame requirement in this function.
        The output are the merged neuron masks and their statistics 
        (acitve frame indices, areas, whether satisfy consecutive activation).

    Inputs: 
        uniques (sparse.csr_matrix): the neuron masks after the first COM merging. 
        times_uniques (list of 1D numpy.array): indices of frames when the neuron is active.
        dims (tuple of int, shape = (2,)): lateral dimension of the image.
        Params_post (dict): Parameters for post-processing.
            Params['thresh_mask']: Threashold to binarize the real-number mask.
            Params['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params['thresh_COM']: Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.

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
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']

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
            # Since this function is used for parameter optimization, searching "cons" will be
            # done in the next step. Here, we just assume all masks are valid neurons.
            have_cons=np.ones(len(masks_final_2), dtype='bool')
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


# %%
def optimize_combine_1_online(list_uniques: list, list_times_uniques: list, dims: tuple, Params: dict, filename_GT: str):
    '''Optimize 1 online post-processing parameter: "cons". 
        Start after the first COM merging.
        The outputs are the recall, precision, and F1 calculated using all values in "list_cons".

    Inputs: 
        list_uniques (list of sparse.csr_matrix of float32, shape = (merge_every,Lx*Ly)): the neuron masks to be merged.
        list_times_uniques (list of list of 1D numpy.array, shape = (merge_every,)): indices of frames when the neuron is active.
        dims (tuple of int, shape = (2,)): the lateral shape of the image.
        Params (dict): Ranges of post-processing parameters to optimize over.
            Params['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params['thresh_COM']: (float or int) Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params['thresh_IOU']: (float) Threshold of IoU used for merging neurons.
            Params['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
            The GT masks are stored in a ".mat" file, and dataset "GTMasks_2" is the GT masks
            (shape = (Ly0*Lx0,n) when saved in MATLAB).

    Outputs:
        Recall_k (1D numpy.array of float): Recall for all cons. 
        Precision_k (1D numpy.array of float): Precision for all cons. 
        F1_k (1D numpy.array of float): F1 for all cons. 
    '''
    thresh_mask = Params['thresh_mask']
    list_cons = Params['list_cons']

    # merge the initalization frames
    tuple_temp = merge_complete_nocons(list_uniques[0], list_times_uniques[0], dims, Params)
    for ind in range(1, len(list_uniques)): 
        # merge additional "merge_every" frames
        tuple_add = merge_complete_nocons(list_uniques[ind], list_times_uniques[ind], dims, Params)
        # merge the new masks with the existing masks
        tuple_temp = merge_2(tuple_temp, tuple_add, dims, Params)
    masks_final_2 = sparse.vstack(tuple_temp[1])
    times_final = tuple_temp[2]

    data_GT=loadmat(filename_GT)
    GTMasks_2 = data_GT['GTMasks_2'].transpose()
    # Search for optimal "cons" used to refine segmented neurons.
    Recall_k, Precision_k, F1_k = refine_seperate_multi(GTMasks_2, \
        masks_final_2, times_final, list_cons, thresh_mask, display=False)
    return Recall_k, Precision_k, F1_k


def optimize_combine_3_online(list_totalmasks, list_neuronstate, list_COMs, list_areas, \
        list_probmapID, dims, minArea, avgArea, Params_set: dict, filename_GT: str, useMP=True):
    '''Optimize 3 online post-processing parameters: "thresh_COM", "thresh_IOU", "cons". 
        Start before the first COM merging, which can include disgarding masks smaller than "minArea".
        The outputs are the recall, precisoin, and F1 calculated from all parameter combinations.

    Inputs: 
        list_totalmasks (list of sparse.csr_matrix of float32, shape = (merge_every,Lx*Ly)): the neuron masks to be merged.
        list_neuronstate (list of 1D numpy.array of bool, shape = (merge_every,)): Indicators of whether a neuron is obtained without watershed.
        list_COMs (list of 2D numpy.array of float, shape = (merge_every,2)): COMs of the neurons.
        list_areas (list of 1D numpy.array of uint32, shape = (merge_every,)): Areas of the neurons. 
        list_probmapID (list of 1D numpy.array of uint32, shape = (merge_every,): indices of frames when the neuron is active. 
        dims (tuple of int, shape = (2,)): the lateral shape of the region.
        minArea (float or int, default to 0): Minimum area of a valid neuron mask (unit: pixels).
        avgArea (float or int, default to 0): The typical neuron area (unit: pixels). 
            Neuron masks with areas larger than avgArea will be further segmented by watershed.
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float or int) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
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

    list_uniques = []
    list_times_uniques = []
    for ind in range(len(list_areas)):
        # Merge neurons with close COM for every merge_every frames.
        uniques, times_uniques = unique_neurons2_simp(list_totalmasks[ind], list_neuronstate[ind], \
            list_COMs[ind], list_areas[ind], list_probmapID[ind], minArea, thresh_COM0)
        list_uniques.append(uniques)
        list_times_uniques.append(times_uniques)

    if not times_uniques:
        list_Recall_inter = np.zeros(size_inter)
        list_Precision_inter = np.zeros(size_inter)
        list_F1_inter = np.zeros(size_inter)
    else:
        # Calculate accuracy scores for various "thresh_COM", "thresh_IOU", and "cons".
        if useMP:
            p3 = mp.Pool()
            list_temp = p3.starmap(optimize_combine_1_online, [(list_uniques, list_times_uniques, dims, 
                {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                'thresh_IOU': thresh_IOU, 'thresh_consume': (1+thresh_IOU)/2, 'cons': 1, 'list_cons':list_cons}, filename_GT) \
                    for thresh_COM in list_thresh_COM for thresh_IOU in list_thresh_IOU], chunksize=1)
            list_Recall_inter = np.vstack([x[0] for x in list_temp]).reshape(size_inter)
            list_Precision_inter = np.vstack([x[1] for x in list_temp]).reshape(size_inter)
            list_F1_inter = np.vstack([x[2] for x in list_temp]).reshape(size_inter)
            p3.close()
            p3.join()
        else: 
            list_Recall_inter = np.zeros(size_inter)
            list_Precision_inter = np.zeros(size_inter)
            list_F1_inter = np.zeros(size_inter)
            for (j1,thresh_COM) in enumerate(list_thresh_COM):
                for (j2,thresh_IOU) in enumerate(list_thresh_IOU):
                    (Recal_k, Precision_k, F1_k) = optimize_combine_1_online(list_uniques, list_times_uniques, dims, 
                        {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                            'thresh_IOU': thresh_IOU, 'thresh_consume': (1+thresh_IOU)/2, 'cons': 1, 'list_cons':list_cons}, filename_GT)                                 
                    list_Recall_inter[j1,j2,:]=Recal_k
                    list_Precision_inter[j1,j2,:]=Precision_k
                    list_F1_inter[j1,j2,:]=F1_k

    return list_Recall_inter, list_Precision_inter, list_F1_inter

    
# %%
def parameter_optimization_online(pmaps: np.ndarray, frames_initf, merge_every, \
        Params_set: dict, filename_GT: str, useMP=True, useWT=False, p=None):
    '''Optimize 6 online post-processing parameters: 
        "minArea", "avgArea", "thresh_pmap", "thresh_COM", "thresh_IOU", "cons". 
        The outputs are the recall, precisoin, and F1 calculated from all parameter combinations.

    Inputs: 
        pmaps (3D numpy.ndarray of uint8, shape = (nframes,Lx,Ly)): the probability map obtained after CNN inference.
            It should not be be previously thresholded. if "thresh_pmap" is going to be optimized
        frames_initf (int): Number of intialization frames after temporal filtering. 
        merge_every (int): SUNS online merge the newly segmented frames every "merge_every" frames.
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float or int) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
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

            num_neurons = np.hstack([x[1] for x in segs2]).size
            if num_neurons==0:
                list_Recall[:,i2,i3,:,:,:]=0
                list_Precision[:,i2,i3,:,:,:]=0
                list_F1[:,i2,i3,:,:,:]=0
            else: 
                # Divide segs into small sub-lists
                # The first sub-list contains the first "framesf" elements
                segs_init = segs2[:frames_initf]
                list_totalmasks = [sparse.vstack([x[0] for x in segs_init])]
                list_neuronstate = [np.hstack([x[1] for x in segs_init])]
                list_COMs = [np.vstack([x[2] for x in segs_init])]
                list_areas = [np.hstack([x[3] for x in segs_init])]
                list_probmapID = [np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs_init)])]
                # Each additional sub-list contains "merge_every" elements
                for t in range(frames_initf, len(segs2), merge_every):
                    segs_add = segs2[t:t+merge_every]
                    list_totalmasks.append(sparse.vstack([x[0] for x in segs_add]))
                    list_neuronstate.append(np.hstack([x[1] for x in segs_add]))
                    list_COMs.append(np.vstack([x[2] for x in segs_add]))
                    list_areas.append(np.hstack([x[3] for x in segs_add]))
                    list_probmapID.append(np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs_add)]))
        
                # Calculate accuracy scores for various "minArea", "thresh_COM", "thresh_IOU", and "cons".
                # "avgArea" and "thresh_pmap" are fixed at this step, because they are searched in outer loops.
                if useMP:
                    try:
                        list_result = p.starmap(optimize_combine_3_online, [(list_totalmasks, \
                            list_neuronstate, list_COMs, list_areas, list_probmapID, dims, minArea, \
                            avgArea, Params_set, filename_GT, False) for minArea in list_minArea])
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
                        list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_3_online(
                            list_totalmasks, list_neuronstate, list_COMs, list_areas, list_probmapID, \
                            dims, minArea, avgArea, Params_set, filename_GT, useMP=False)
                        list_Recall[i1,i2,i3,:,:,:]=list_Recall_inter
                        list_Precision[i1,i2,i3,:,:,:]=list_Precision_inter
                        list_F1[i1,i2,i3,:,:,:]=list_F1_inter
                        print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[i1,i2,i3,:,:,:].max()))

    if useMP and closep:
        p.close()
        p.join()
    return list_Recall, list_Precision, list_F1

