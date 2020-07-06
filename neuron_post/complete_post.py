import io
import numpy as np
from scipy import sparse
from scipy import signal
from scipy.io import savemat, loadmat
import time
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

from seperate_multi import separateNeuron_noWT, watershed_neurons, separateNeuron, separateNeuron_b
# from seperate_fast import separateNeuron
from combine import uniqueNeurons1_simp, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume
from evaluate_post import refine_seperate_nommin, refine_seperate_multi
from par3 import fastmovmean


# %%
def complete_segment(pmaps: np.ndarray, Params: dict, useMP=True, useWT=False, useMATLAB=False, display=False, eng=None, p=None):
    '''The input probablity map must be pre-thresholded'''
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    # nframes = len(pmaps)
    # (Lx, Ly) = pmaps[0].shape
    minArea = Params['minArea']
    avgArea = Params['avgArea']
    thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    # win_avg = Params['win_avg']
    # if useMP: # %% Run segmentation with multiprocessing
    #     p = mp.Pool(mp.cpu_count())
    start_all = time.time()

    # if win_avg > 1:
    #     start = time.time()
    #     pmaps_avg = np.zeros(dims, dtype = pmaps.dtype)
    #     if useMP:
    #         fastmovmean(pmaps, pmaps_avg, win_avg)
    #     else:
    #         before = win_avg//2
    #         after = win_avg-before
    #         for kk in range(nframes):
    #             pmaps_avg [kk] = pmaps[max(0,kk-before):min(nframes,kk+after)].mean(axis = 0)
    #     pmaps = pmaps_avg
    #     print('Move mean filter: {} s'.format(time.time() - start))

    # useMP=False
    # p.close()
    if useMP: # %% Run segmentation with multiprocessing
        start = time.time()
        segs = p.starmap(separateNeuron_b, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
        end = time.time()
    else: # %% Run segmentation without multiprocessing
        segs = []
        start = time.time()
        for ind in range(nframes):
            segs.append(separateNeuron_b(pmaps[ind], thresh_pmap, minArea, avgArea, useWT)) #, eng
        end = time.time()
    num_neurons = sum([x[1].size for x in segs])
    if display:
        print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
            .format('separate Neurons', end-start,(end-start)/nframes*1000),
                '{:6d} segmented neurons.'.format(num_neurons))

    if num_neurons==0:
        print('No masks found. Please lower minArea or thresh_pmap.')
        Masks_2 = sparse.csc_matrix((0,Lx*Ly), dtype='bool')
    else:
        if useMATLAB: # %% Do the combination using MATLAB
            # eng = engine.start_matlab() # '-desktop'
            # eng.gcp()
            eng = engine.connect_matlab()
            start = time.time()
            totalmasks = sparse.vstack([x[0] for x in segs])
            neuronstate = np.hstack([x[1] for x in segs])
            COMs = np.vstack([x[2] for x in segs])
            probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
            
            filename_input='python_output.mat'
            filename_output='MATLAB_output.mat'
            data_dict={'totalmasks': totalmasks, 'neuronstate': neuronstate, 'COMs': COMs, 'probmapID': probmapID,
                    'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_mask': thresh_mask, 'AvgArea': avgArea,
                    'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'Length': (dims[1], dims[2])}
            savemat(filename_input, data_dict)
            out = io.StringIO()
            err = io.StringIO()
            eng.combine_neurons_from_py(filename_input, filename_output, nargout=0, stdout=out, stderr=err)
            print(out.getvalue())
            print(err.getvalue())
            eng.exit()        
            if display:
                end = time.time()
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame'\
                    .format('Combine neurons', end - start, (end - start) / nframes * 1000))

            matlab_result=loadmat(filename_output)
            masks_final_2=matlab_result['masks_final_2'].transpose()
            times_final=(matlab_result['times_final']).tolist()   
            times_final=[x[0].squeeze()-1 for x in times_final]

        else: # %% Do the combination in Python
            start = time.time()
            # %% uniqueNeurons1
            # uniques, times_uniques = uniqueNeurons1(segs, thresh_COM0)
            uniques, times_uniques = uniqueNeurons1_simp(segs, thresh_COM0) # minArea, , p
            end_unique = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('uniqueNeurons1', end_unique - start, (end_unique - start) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_uniques)))
            # %% group_neurons
            groupedneurons, times_groupedneurons = \
                group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
            end_COM = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('group_neurons', end_COM - end_unique, (end_COM - end_unique) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_groupedneurons)))
            # %% piece_neurons
            piecedneurons_1, times_piecedneurons_1 = \
                piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
            end_IOU = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('piece_neurons_IOU', end_IOU - end_COM, (end_IOU - end_COM) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_piecedneurons_1)))
            piecedneurons, times_piecedneurons = \
                piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
            end_consume = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('piece_neurons_consume', end_consume - end_IOU, (end_consume - end_IOU) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_piecedneurons)))
            # %% Final result
            masks_final_2 = piecedneurons
            times_final = [np.unique(x) for x in times_piecedneurons]
            
        # %% Refine neurons using consecutive occurence
        start = time.time()
        Masks_2 = refine_seperate_nommin(masks_final_2, times_final, cons, thresh_mask)
        end_all = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('refine_seperate', end_all - start, (end_all - start) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('Total time', end_all - start_all, (end_all - start_all) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))

    # if useMP:
    #     p.close()
    #     p.join()
    return Masks_2 # masks_final_2, times_final #, matname_output


def complete_segment_pre(pmaps: np.ndarray, Params: dict, useMP=True, useWT=False, useMATLAB=False, display=False, eng=None, p=None):
    dims=pmaps.shape
    (nframes, Lx, Ly) = dims
    # nframes = len(pmaps)
    # (Lx, Ly) = pmaps[0].shape
    minArea = Params['minArea']
    avgArea = Params['avgArea']
    thresh_pmap = Params['thresh_pmap']
    thresh_mask = Params['thresh_mask']
    thresh_COM0 = Params['thresh_COM0']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = Params['thresh_consume']
    cons = Params['cons']
    win_avg = Params['win_avg']
    # if useMP: # %% Run segmentation with multiprocessing
    #     p = mp.Pool(mp.cpu_count())
    start_all = time.time()

    if win_avg > 1:
        start = time.time()
        pmaps_avg = np.zeros(dims, dtype = pmaps.dtype)
        if useMP:
            fastmovmean(pmaps, pmaps_avg, win_avg)
        else:
            before = win_avg//2
            after = win_avg-before
            for kk in range(nframes):
                pmaps_avg [kk] = pmaps[max(0,kk-before):min(nframes,kk+after)].mean(axis = 0)
        pmaps = pmaps_avg
        print('Move mean filter: {} s'.format(time.time() - start))

    # useMP=False
    # p.close()
    if useMP: # %% Run segmentation with multiprocessing
        start = time.time()
        segs = p.starmap(separateNeuron, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
        end = time.time()
    else: # %% Run segmentation without multiprocessing
        segs = []
        start = time.time()
        for ind in range(nframes):
            segs.append(separateNeuron(pmaps[ind], thresh_pmap, minArea, avgArea, useWT)) #, eng
        end = time.time()
    num_neurons = sum([x[1].size for x in segs])
    if display:
        print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
            .format('separate Neurons', end-start,(end-start)/nframes*1000),
                '{:6d} segmented neurons.'.format(num_neurons))

    if num_neurons==0:
        print('No masks found. Please lower minArea or thresh_pmap.')
        Masks_2 = sparse.csc_matrix((0,Lx*Ly), dtype='bool')
    else:
        if useMATLAB: # %% Do the combination using MATLAB
            # eng = engine.start_matlab() # '-desktop'
            # eng.gcp()
            eng = engine.connect_matlab()
            start = time.time()
            totalmasks = sparse.vstack([x[0] for x in segs])
            neuronstate = np.hstack([x[1] for x in segs])
            COMs = np.vstack([x[2] for x in segs])
            probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
            
            filename_input='python_output.mat'
            filename_output='MATLAB_output.mat'
            data_dict={'totalmasks': totalmasks, 'neuronstate': neuronstate, 'COMs': COMs, 'probmapID': probmapID,
                    'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_mask': thresh_mask, 'AvgArea': avgArea,
                    'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'Length': (dims[1], dims[2])}
            savemat(filename_input, data_dict)
            out = io.StringIO()
            err = io.StringIO()
            eng.combine_neurons_from_py(filename_input, filename_output, nargout=0, stdout=out, stderr=err)
            print(out.getvalue())
            print(err.getvalue())
            eng.exit()        
            if display:
                end = time.time()
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame'\
                    .format('Combine neurons', end - start, (end - start) / nframes * 1000))

            matlab_result=loadmat(filename_output)
            masks_final_2=matlab_result['masks_final_2'].transpose()
            times_final=(matlab_result['times_final']).tolist()   
            times_final=[x[0].squeeze()-1 for x in times_final]

        else: # %% Do the combination in Python
            start = time.time()
            # %% uniqueNeurons1
            # uniques, times_uniques = uniqueNeurons1(segs, thresh_COM0)
            uniques, times_uniques = uniqueNeurons1_simp(segs, thresh_COM0) # minArea, , p
            end_unique = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('uniqueNeurons1', end_unique - start, (end_unique - start) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_uniques)))
            # %% group_neurons
            groupedneurons, times_groupedneurons = \
                group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
            end_COM = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('group_neurons', end_COM - end_unique, (end_COM - end_unique) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_groupedneurons)))
            # %% piece_neurons
            piecedneurons_1, times_piecedneurons_1 = \
                piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
            end_IOU = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('piece_neurons_IOU', end_IOU - end_COM, (end_IOU - end_COM) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_piecedneurons_1)))
            piecedneurons, times_piecedneurons = \
                piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
            end_consume = time.time()
            if display:
                print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                    .format('piece_neurons_consume', end_consume - end_IOU, (end_consume - end_IOU) / nframes * 1000),\
                        '{:6d} segmented neurons.'.format(len(times_piecedneurons)))
            # %% Final result
            masks_final_2 = piecedneurons
            times_final = [np.unique(x) for x in times_piecedneurons]
            
        # %% Refine neurons using consecutive occurence
        start = time.time()
        Masks_2 = refine_seperate_nommin(masks_final_2, times_final, cons, thresh_mask)
        end_all = time.time()
        if display:
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('refine_seperate', end_all - start, (end_all - start) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))
            print('{:25s}: Used {:9.6f} s, {:9.6f} ms/frame, '\
                .format('Total time', end_all - start_all, (end_all - start_all) / nframes * 1000),\
                    '{:6d} segmented neurons.'.format(len(times_final)))

    # if useMP:
    #     p.close()
    #     p.join()
    return Masks_2 # masks_final_2, times_final #, matname_output


# %%
def optimize_combine(uniques: sparse.csr_matrix, times_uniques: list, dims: tuple, Params: dict, filename_GT: str):  #, avgArea, thresh_mask, thresh_COM, thresh_IOU, thresh_consume, list_mmin, list_cons #, filename_GT_dilate=None, eng
    avgArea = Params['avgArea']
    thresh_mask = Params['thresh_mask']
    thresh_COM = Params['thresh_COM']
    thresh_IOU = Params['thresh_IOU']
    thresh_consume = (1+thresh_IOU)/2
    # list_thresh_consume = Params_set['list_thresh_consume']
    list_cons = Params['list_cons'] #= list(range(1, 13))
    # L_thresh_COM=len(list_thresh_COM)
    # L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)

    groupedneurons, times_groupedneurons = group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
    piecedneurons_1, times_piecedneurons_1 = piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
    piecedneurons, times_piecedneurons = piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
    masks_final_2 = piecedneurons
    times_final = [np.unique(x) for x in times_piecedneurons]

    # masks_final_2_mat = masks_final_2.transpose()
    # times_final_mat = [np.expand_dims(x+1, axis=1) for x in times_final]
    # matname_output = '.\\Python results\\Python result, COM={:0.1f}.mat'.format(thresh_COM)
    # savemat(matname_output, {'masks_final_2': masks_final_2_mat, 'times_final': times_final_mat})

    data_GT=loadmat(filename_GT)
    GTMasks_2 = data_GT['GTMasks_2'].transpose()
    Recall_k, Precision_k, F1_k = refine_seperate_multi(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask, display=False)
    # Recall_k, Precision_k, F1_k = evaluate_segmentation_MATLAB(matname_output, filename_GT, list_cons, list_mmin, thresh_mask, False) # eng, 
    return Recall_k, Precision_k, F1_k


def optimize_combine_minArea(totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set: dict, filename_GT: str, useMP=True):
    thresh_mask = Params_set['thresh_mask']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    list_cons = Params_set['list_cons']
    
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_cons=len(list_cons)
    size_inter = (L_thresh_COM, L_thresh_IOU, L_cons)
    # print('Using minArea={}, avgArea={}'.format(minArea, avgArea))

    uniques, times_uniques = uniqueNeurons2_simp(totalmasks, neuronstate, COMs, areas, probmapID, minArea, thresh_COM0)

    if not times_uniques:
        list_Recall_inter = np.zeros(size_inter)
        list_Precision_inter = np.zeros(size_inter)
        list_F1_inter = np.zeros(size_inter)
    else:
        # # %% Do the combination in Python
        if useMP:
            p3 = mp.Pool()
            list_temp = p3.starmap(optimize_combine, [(uniques, times_uniques, dims, 
                {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                'thresh_IOU': thresh_IOU, 'list_cons':list_cons}, filename_GT) \
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
                    thresh_consume = (1+thresh_IOU)/2
                    (Recal_k, Precision_k, F1_k) = optimize_combine(uniques, times_uniques, dims, 
                        {'avgArea': avgArea, 'thresh_mask': thresh_mask, 'thresh_COM': thresh_COM,
                            'thresh_IOU': thresh_IOU, 'list_cons':list_cons}, filename_GT)                                 
                    list_Recall_inter[j1,j2,:]=Recal_k
                    list_Precision_inter[j1,j2,:]=Precision_k
                    list_F1_inter[j1,j2,:]=F1_k

    return list_Recall_inter, list_Precision_inter, list_F1_inter

    
# %%
def paremter_optimization_before(pmaps: np.ndarray, Params_set: dict, filename_GT, useMP=True, useWT=False, eng=None):
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
    list_win_avg = Params_set['list_win_avg']
    list_cons = [1]

    L_minArea=len(list_minArea)
    L_avgArea=len(list_avgArea)
    L_thresh_pmap=len(list_thresh_pmap)
    L_thresh_COM=len(list_thresh_COM)
    L_thresh_IOU=len(list_thresh_IOU)
    L_win_avg=len(list_win_avg)
    dim_result = (L_minArea, L_avgArea, L_thresh_pmap, L_win_avg, L_thresh_COM, L_thresh_IOU)
    list_Recall = np.zeros(dim_result)
    list_Precision = np.zeros(dim_result)
    list_F1 = np.zeros(dim_result)
    if useMP:
        p = mp.Pool(mp.cpu_count())
        p2 = mp.Pool(L_minArea)
    start = time.time()

    for (i4, win_avg) in enumerate(list_win_avg):
        print('Using win_avg={}'.format(win_avg))
        if win_avg == 1:
            pmaps_avg = pmaps
        else:
            pmaps_avg = np.zeros(dims, dtype = pmaps.dtype)
            if useMP:
                fastmovmean(pmaps, pmaps_avg, win_avg)
            else:
                before = win_avg//2
                after = win_avg-before
                for kk in range(nframes):
                    pmaps_avg [kk] = pmaps[max(0,kk-before):min(nframes,kk+after)].mean(axis = 0)
            print('Used {} s'.format(time.time() - start))

        for (i3,thresh_pmap) in enumerate(list_thresh_pmap):
            print('Using thresh_pmap={}, win_avg={}'.format(thresh_pmap, win_avg))
            minArea = min(list_minArea)
            if useMP: # %% Run segmentation with multiprocessing
                segs = p.starmap(separateNeuron_noWT, [(frame, thresh_pmap, minArea) for frame in pmaps_avg], chunksize=1) #, eng
            else: # %% Run segmentation without multiprocessing
                segs = [separateNeuron_noWT(frame, thresh_pmap, minArea) for frame in pmaps_avg]
            print('Used {} s'.format(time.time() - start))

            for (i2,avgArea) in enumerate(list_avgArea):
                if useWT:
                    print('Using avgArea={}, thresh_pmap={}, win_avg={}'.format(avgArea, thresh_pmap, win_avg))
                    if useMP: # %% Run segmentation with multiprocessing
                        segs2 = p.starmap(watershed_neurons, [((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs], chunksize=32) #, eng
                    else: # %% Run segmentation without multiprocessing
                        segs2 = [watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs]
                    print('Used {} s'.format(time.time() - start))
                else:
                    segs2 = segs # for no watershed

                totalmasks = sparse.vstack([x[0] for x in segs2])
                neuronstate = np.hstack([x[1] for x in segs2])
                COMs = np.vstack([x[2] for x in segs2])
                areas = np.hstack([x[3] for x in segs2])
                probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs2)])

                num_neurons = neuronstate.size
                if num_neurons==0:
                    list_Recall[:,i2,i3,i4,:,:]=0
                    list_Precision[:,i2,i3,i4,:,:]=0
                    list_F1[:,i2,i3,i4,:,:]=0
                else:
                    if useMP:
                        list_result = p2.starmap(optimize_combine_minArea, [(totalmasks, neuronstate, COMs, areas, \
                            probmapID, dims, minArea, avgArea, Params_set, filename_GT, False) for minArea in list_minArea])
                        for i1 in range(L_minArea):
                            list_Recall[i1,i2,i3,i4,:,:]=list_result[i1][0].squeeze(axis=-1)
                            list_Precision[i1,i2,i3,i4,:,:]=list_result[i1][1].squeeze(axis=-1)
                            list_F1[i1,i2,i3,i4,:,:]=list_result[i1][2].squeeze(axis=-1)
                        print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[:,i2,i3,i4,:,:].max()))
                    else:
                        for (i1,minArea) in enumerate(list_minArea):
                            print('Using minArea={}, avgArea={}, thresh_pmap={}, win_avg={}'.format(minArea, avgArea, thresh_pmap, win_avg))
                            list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_minArea(
                                totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set, filename_GT, useMP=False)
                            list_Recall[i1,i2,i3,i4,:,:]=list_Recall_inter[0]
                            list_Precision[i1,i2,i3,i4,:,:]=list_Precision_inter[0]
                            list_F1[i1,i2,i3,i4,:,:]=list_F1_inter[0]
                            print('Used {} s, '.format(time.time() - start) + 'Best F1 is {}'.format(list_F1[i1,i2,i3,i4,:,:].max()))
        del pmaps_avg

    if useMP:
        p2.close()
        p2.join()
        p.close()
        p.join()
    list_Recall = np.transpose(list_Recall, [0,1,2,4,5,3])
    list_Precision = np.transpose(list_Precision, [0,1,2,4,5,3])
    list_F1 = np.transpose(list_F1, [0,1,2,4,5,3])
    return list_Recall, list_Precision, list_F1


# %%
def paremter_optimization_after(pmaps: np.ndarray, Params_set: dict, filename_GT: str, useMP=True, useWT=False, eng=None, p=None): #
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
                    for (ind, frame_seg) in enumerate(segs):
                        segs2.append(watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea))
                    # segs2 = [watershed_neurons((Lx, Ly), frame_seg, minArea, avgArea) for frame_seg in segs]
                print('Used {} s'.format(time.time() - start))
            else:
                segs2 = segs # for no watershed

            totalmasks = sparse.vstack([x[0] for x in segs2])
            neuronstate = np.hstack([x[1] for x in segs2])
            COMs = np.vstack([x[2] for x in segs2])
            areas = np.hstack([x[3] for x in segs2])
            probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs2)])

            num_neurons = neuronstate.size
            if num_neurons==0 or totalmasks.nnz/pmaps.size>0.04:
                list_Recall[:,i2,i3,:,:,:]=0
                list_Precision[:,i2,i3,:,:,:]=0
                list_F1[:,i2,i3,:,:,:]=0
            else:
                if useMP:
                    try:
                        list_result = p.starmap(optimize_combine_minArea, [(totalmasks, neuronstate, COMs, areas, \
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
                        print('OverflowError. Size of totalmasks is larger than 4 GB. Thresh_pmap is likely too low.')
                    except MemoryError: 
                        list_Recall[:,i2,i3,:,:,:]=0
                        list_Precision[:,i2,i3,:,:,:]=0
                        list_F1[:,i2,i3,:,:,:]=0
                        print('MemoryError. Too much memory is needed. Thresh_pmap is likely too low.')

                else:
                    for (i1,minArea) in enumerate(list_minArea):
                        print('Using minArea={}, avgArea={}, thresh_pmap={}'.format(minArea, avgArea, thresh_pmap))
                        list_Recall_inter, list_Precision_inter, list_F1_inter = optimize_combine_minArea(
                            totalmasks, neuronstate, COMs, areas, probmapID, dims, minArea, avgArea, Params_set, filename_GT, useMP=False)
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
