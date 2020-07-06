import sys
import io
import numpy as np
from scipy import sparse
from scipy import signal
from scipy.io import savemat, loadmat
import time
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

import functions_online
import functions_init
sys.path.insert(1, '..\\neuron_post')
from seperate_multi import separateNeuron_noWT, watershed_neurons, separateNeuron, separateNeuron_b
# from seperate_fast import separateNeuron
from combine import uniqueNeurons1_simp, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume
from evaluate_post import refine_seperate_nommin, refine_seperate_multi
from par3 import fastmovmean


def merge_complete_2(uniques, times_uniques, dims, Params): # , select_cons=True
    # minArea = Params['minArea']
    avgArea = Params['avgArea']
    # thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    # thresh_consume = (1+thresh_IOU)/2
    cons = Params['cons']
    # cons = Params['cons']
    # win_avg = Params['win_avg']

    # uniques, times_uniques = uniqueNeurons1_simp(segs, thresh_COM0) # minArea,
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
        # masks_2_float = refine_seperate_nommin_float(masks_final_2, times_final, cons, thresh_mask)
        # masks_2_float = masks_final_2
        if masks_final_2.size:
            masks_final_2 = [x for x in masks_final_2]
            Masks_2 = [(x >= x.max() * thresh_mask).astype('float') for x in masks_final_2]
            area = np.array([x.nnz for x in Masks_2]) # Masks_2.sum(axis=1).A.squeeze()
            # if select_cons:
            #     have_cons = refine_seperate_cons(times_final, cons)
            # else:
            have_cons=np.ones(len(masks_final_2), dtype='bool')
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


# %%
def optimize_combine_online(list_uniques: list, list_times_uniques: list, dims: tuple, Params: dict, filename_GT: str):  #, avgArea, thresh_mask, thresh_COM, thresh_IOU, thresh_consume, list_mmin, list_cons #, filename_GT_dilate=None, eng
    # avgArea = Params['avgArea']
    thresh_mask = Params['thresh_mask']
    # thresh_COM = Params['thresh_COM']
    # thresh_IOU = Params['thresh_IOU']
    # thresh_consume = (1+thresh_IOU)/2
    # thresh_consume = Params['thresh_consume']
    # cons = 1
    # list_thresh_consume = Params_set['list_thresh_consume']
    list_cons = Params['list_cons'] #= list(range(1, 13))
    # L_thresh_COM=len(list_thresh_COM)
    # L_thresh_IOU=len(list_thresh_IOU)
    # L_cons=len(list_cons)

    tuple_temp = merge_complete_2(list_uniques[0], list_times_uniques[0], dims, Params)
    for ind in range(1, len(list_uniques)):
        tuple_add = merge_complete_2(list_uniques[ind], list_times_uniques[ind], dims, Params)
        tuple_temp = functions_online.merge_2(tuple_temp, tuple_add, dims, Params)
        # groupedneurons, times_groupedneurons = group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
        # piecedneurons_1, times_piecedneurons_1 = piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
        # piecedneurons, times_piecedneurons = piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
        # masks_final_2 = piecedneurons
        # times_final = [np.unique(x) for x in times_piecedneurons]
    # masks_final_2_mat = masks_final_2.transpose()
    # times_final_mat = [np.expand_dims(x+1, axis=1) for x in times_final]
    # matname_output = '.\\Python results\\Python result, COM={:0.1f}.mat'.format(thresh_COM)
    # savemat(matname_output, {'masks_final_2': masks_final_2_mat, 'times_final': times_final_mat})
    masks_final_2 = sparse.vstack(tuple_temp[1])
    times_final = tuple_temp[2]

    data_GT=loadmat(filename_GT)
    GTMasks_2 = data_GT['GTMasks_2'].transpose()
    Recall_k, Precision_k, F1_k = refine_seperate_multi(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask, display=False)
    # Recall_k, Precision_k, F1_k = evaluate_segmentation_MATLAB(matname_output, filename_GT, list_cons, list_mmin, thresh_mask, False) # eng, 
    return Recall_k, Precision_k, F1_k


def optimize_combine_minArea_online(list_totalmasks, list_neuronstate, list_COMs, list_areas, \
        list_probmapID, dims, minArea, avgArea, Params_set: dict, filename_GT: str, useMP=True):
    thresh_mask = Params_set['thresh_mask']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    # thresh_consume = Params_set['thresh_consume']
    list_cons = Params_set['list_cons']

    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    size_inter = (L_thresh_COM, L_thresh_IOU, L_cons)
    # print('Using minArea={}, avgArea={}'.format(minArea, avgArea))

    list_uniques = []
    list_times_uniques = []
    for ind in range(len(list_areas)):
        uniques, times_uniques = uniqueNeurons2_simp(list_totalmasks[ind], list_neuronstate[ind], \
            list_COMs[ind], list_areas[ind], list_probmapID[ind], minArea, thresh_COM0)
        list_uniques.append(uniques)
        list_times_uniques.append(times_uniques)

    if not times_uniques:
        list_Recall_inter = np.zeros(size_inter)
        list_Precision_inter = np.zeros(size_inter)
        list_F1_inter = np.zeros(size_inter)
    else:
        # # %% Do the combination in Python
        if useMP:
            p3 = mp.Pool()
            list_temp = p3.starmap(optimize_combine_online, [(list_uniques, list_times_uniques, dims, 
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
                    # thresh_consume = (1+thresh_IOU)/2
                    (Recal_k, Precision_k, F1_k) = optimize_combine_online(list_uniques, list_times_uniques, dims, 
                        {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                            'thresh_IOU': thresh_IOU, 'thresh_consume': (1+thresh_IOU)/2, 'cons': 1, 'list_cons':list_cons}, filename_GT)                                 
                    list_Recall_inter[j1,j2,:]=Recal_k
                    list_Precision_inter[j1,j2,:]=Precision_k
                    list_F1_inter[j1,j2,:]=F1_k

    return list_Recall_inter, list_Precision_inter, list_F1_inter

    
# %%
def paremter_optimization_after_online(pmaps: np.ndarray, frames_initf, merge_every, \
        Params_set: dict, filename_GT: str, useMP=True, useWT=False, p=None): #
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    # nframes = len(pmaps)
    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    # thresh_mask = Params_set['thresh_mask']
    # thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    # list_thresh_consume = Params_set['list_thresh_consume']
    list_cons = Params_set['list_cons'] #= list(range(1, 13))

    L_minArea=len(list_minArea)
    L_avgArea=len(list_avgArea)
    L_thresh_pmap=len(list_thresh_pmap)
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    dim_result = (L_minArea, L_avgArea, L_thresh_pmap, L_thresh_COM, L_thresh_IOU, L_cons)
    list_Recall = np.zeros(dim_result)
    list_Precision = np.zeros(dim_result)
    list_F1 = np.zeros(dim_result)
    # size_inter = (L_thresh_COM, L_thresh_IOU, L_cons)
    # useMP=False

    if useMP and not p: #
        p = mp.Pool(mp.cpu_count())
        closep = True
    else:
        closep = False
        # p2 = mp.Pool(L_minArea)
        # filename_output='MATLAB_output.mat'
    start = time.time()

    for (i3,thresh_pmap) in enumerate(list_thresh_pmap):
        # if thresh_pmap<190:
        #     continue
        print('Using thresh_pmap={}'.format(thresh_pmap))
        minArea = min(list_minArea)
        if useMP: # %% Run segmentation with multiprocessing
            segs = p.starmap(separateNeuron_noWT, [(frame, thresh_pmap, minArea) for frame in pmaps], chunksize=1) #, eng
        else: # %% Run segmentation without multiprocessing
            segs = []
            for frame in pmaps:
                segs.append(separateNeuron_noWT(frame, thresh_pmap, minArea))
        print('Used {} s'.format(time.time() - start))

        # useMP=False
        # p.close()
        for (i2,avgArea) in enumerate(list_avgArea):
            if useWT:
                print('Using avgArea={}, thresh_pmap={}'.format(avgArea, thresh_pmap))
                if useMP: # %% Run segmentation with multiprocessing
                    segs2 = p.starmap(watershed_neurons, [((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs], chunksize=32) #, eng
                else: # %% Run segmentation without multiprocessing
                    segs2 = []
                    for frame_seg in segs:
                        segs2.append(watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea))
                    # segs2 = [watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs]
                print('Used {} s'.format(time.time() - start))
            else:
                segs2 = segs # for no watershed

            num_neurons = np.hstack([x[1] for x in segs2]).size
            if num_neurons==0: # or totalmasks.nnz/pmaps.size>0.04:
                list_Recall[:,i2,i3,:,:,:]=0
                list_Precision[:,i2,i3,:,:,:]=0
                list_F1[:,i2,i3,:,:,:]=0
            else:
                segs_init = segs2[:frames_initf]
                list_totalmasks = [sparse.vstack([x[0] for x in segs_init])]
                list_neuronstate = [np.hstack([x[1] for x in segs_init])]
                list_COMs = [np.vstack([x[2] for x in segs_init])]
                list_areas = [np.hstack([x[3] for x in segs_init])]
                list_probmapID = [np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs_init)])]
                for t in range(frames_initf, len(segs2), merge_every):
                    segs_add = segs2[t:t+merge_every]
                    list_totalmasks.append(sparse.vstack([x[0] for x in segs_add]))
                    list_neuronstate.append(np.hstack([x[1] for x in segs_add]))
                    list_COMs.append(np.vstack([x[2] for x in segs_add]))
                    list_areas.append(np.hstack([x[3] for x in segs_add]))
                    list_probmapID.append(np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs_add)]))
        
                if useMP:
                    try:
                        list_result = p.starmap(optimize_combine_minArea_online, [(list_totalmasks, \
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
                        print('OverflowError. Size of totalmasks is larger than 4 GB. Thresh_pmap is likely too low.')
                    except MemoryError: 
                        list_Recall[:,i2,i3,:,:,:]=0
                        list_Precision[:,i2,i3,:,:,:]=0
                        list_F1[:,i2,i3,:,:,:]=0
                        print('MemoryError. Too much memory is needed. Thresh_pmap is likely too low.')

                else:
                    for (i1,minArea) in enumerate(list_minArea):
                        print('Using minArea={}, avgArea={}, thresh_pmap={}'.format(minArea, avgArea, thresh_pmap))
                        list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_minArea_online(
                            list_totalmasks, list_neuronstate, list_COMs, list_areas, list_probmapID, \
                            dims, minArea, avgArea, Params_set, filename_GT, useMP=False)
                        list_Recall[i1,i2,i3,:,:,:]=list_Recall_inter
                        list_Precision[i1,i2,i3,:,:,:]=list_Precision_inter
                        list_F1[i1,i2,i3,:,:,:]=list_F1_inter
                        print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[i1,i2,i3,:,:,:].max()))

    if useMP and closep:
        # p2.close()
        # p2.join()
        p.close()
        p.join()
    return list_Recall, list_Precision, list_F1


# %%
def paremter_optimization_WT_after(pmaps: np.ndarray, Params_set: dict, filename_GT: str, useMP=True, eng=None):
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    # nframes = len(pmaps)
    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    thresh_mask = Params_set['thresh_mask']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    # list_thresh_consume = Params_set['list_thresh_consume']
    list_cons = Params_set['list_cons'] #= list(range(1, 13))

    L_minArea=len(list_minArea)
    L_avgArea=len(list_avgArea)
    L_thresh_pmap=len(list_thresh_pmap)
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    dim_result = (L_minArea, L_avgArea, L_thresh_pmap, L_thresh_COM, L_thresh_IOU, L_cons)
    list_Recall = np.zeros(dim_result)
    list_Precision = np.zeros(dim_result)
    list_F1 = np.zeros(dim_result)
    size_inter = (L_thresh_COM, L_thresh_IOU, L_cons)

    if useMP:
        p = mp.Pool(mp.cpu_count())
        # filename_output='MATLAB_output.mat'
    start = time.time()

    for (i1,minArea) in enumerate(list_minArea):
        for (i2,avgArea) in enumerate(list_avgArea):
            for (i3,thresh_pmap) in enumerate(list_thresh_pmap):
                print('Using minArea={}, avgArea={}, thresh_pmap={}'.format(minArea, avgArea, thresh_pmap))
                if useMP: # %% Run segmentation with multiprocessing
                    segs = p.starmap(separateNeuron, [(frame, thresh_pmap, minArea, avgArea, True) for frame in pmaps], chunksize=1) #, eng
                else: # %% Run segmentation without multiprocessing
                    segs = []
                    for ind in range(nframes):
                        segs.append(separateNeuron(pmaps[ind], thresh_pmap, minArea, avgArea, True))
                print('Used {} s, '.format(time.time() - start))

                totalmasks = sparse.vstack([x[0] for x in segs])
                neuronstate = np.hstack([x[1] for x in segs])
                COMs = np.vstack([x[2] for x in segs])
                areas = np.hstack([x[3] for x in segs])
                probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])

                num_neurons = neuronstate.size
                if num_neurons==0:
                    list_Recall_inter = np.zeros(size_inter)
                    list_Precision_inter = np.zeros(size_inter)
                    list_F1_inter = np.zeros(size_inter)
                else:
                    list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_minArea(
                        totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set, filename_GT, useMP=useMP)

                list_Recall[i1,i2,i3,:,:,:]=list_Recall_inter
                list_Precision[i1,i2,i3,:,:,:]=list_Precision_inter
                list_F1[i1,i2,i3,:,:,:]=list_F1_inter
                print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[i1,i2,i3,:,:,:].max()))

    if useMP:
        p.close()
        p.join()
    return list_Recall, list_Precision, list_F1
