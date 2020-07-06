# %%
import os
import cv2
import math
import numpy as np
# import matplotlib.pyplot as plt
import time
import h5py
import sys
import pyfftw
from scipy import sparse

# import random
# import tensorflow as tf
from scipy.io import savemat, loadmat
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import par1
# from preprocessing_functions import process_video, process_video_prealloc
# from par2 import fastuint, fastcopy
# from par3 import fastthreshold
from unet4_best import get_unet
from evaluate_post import GetPerformance_Jaccard_2
# from complete_post import complete_segment
import functions_online
import functions_init
import par_online
from seperate_multi import separateNeuron_b
from combine import uniqueNeurons1_simp, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume


# %%
def main():
    list_Exp_ID = [ '501484643','501574836','501729039','502608215','503109347',
                    '510214538','524691284','527048992','531006860','539670003']
    thred_std = 7
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    batch_size_eval = 1
    useSF=True
    useTF=True
    useSNR=True
    useWT=False
    show_intermediate=True
    merge_every = 30
    frames_init = 900
    batch_size_init = 180

    dir_video = 'D:\\ABO\\20 percent\\'
    dir_parent = dir_video + 'ShallowUNet\\complete\\'
    # dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+20BCE\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+100FL(1,0.25)\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_output = dir_parent + dir_sub + 'output_masks online video\\' #.format(merge_every)
    dir_params = dir_parent + dir_sub + 'output_masks\\'
    weights_path = dir_parent + dir_sub + 'Weights\\'
    dir_GTMasks = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 

    # %% PreProcessing parameters
    nn = 23200 # nframes # cv2.getOptimalDFTSize(nframes)
    gauss_filt_size = 50  # signa in pixels
    # num_median_approx = 1000
    # network_baseline = 0 # 64.0
    # network_SNRscale = 1 # 32.0

    # frame_rate = 30
    # decay = 0.2
    # leng_tf = math.ceil(frame_rate*decay) # 6
    if useTF:
        h5f = h5py.File(r'C:\Matlab Files\PreProcessing\GCaMP6f_spike_tempolate_mean.h5','r')
        Poisson_filt = np.array(h5f['filter_tempolate']).squeeze()[3:12].astype('float32')
    else:
        Poisson_filt = np.array([1], dtype='float32')
    leng_tf = Poisson_filt.size
    leng_past = 2*leng_tf
    # Params_pre = {'gauss_filt_size':gauss_filt_size, #'num_median_approx':num_median_approx, 
    #     # 'network_baseline':network_baseline, 'network_SNRscale':network_SNRscale, 
    #     'nn':frames_init, 'Poisson_filt': Poisson_filt}
        # 'frame_rate':frame_rate, 'decay':decay, 'leng_tf':leng_tf, 

    # %% Network parameters
    # num_train_per = 200
    size = 488
    # rows = cols = size

    # %% PostProcessing parameters
    Lx = Ly = 487
    dims = (Lx,Ly)
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)
    p = mp.Pool() #mp.cpu_count()

    list_CV = list(range(0,10))
    num_CV = len(list_CV)
    list_Recall = np.zeros((num_CV, 1))
    list_Precision = np.zeros((num_CV, 1))
    list_F1 = np.zeros((num_CV, 1))
    list_time = np.zeros((num_CV, 3))
    list_time_frame = np.zeros((num_CV, 3))

    for CV in [0:1]: # list_CV: # 
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)
        list_time_per = np.zeros(nn)
        start = time.time()

        # Load CNN models
        fff = get_unet() #size
        fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))
        init_imgs = np.zeros((batch_size_eval, size, size, 1), dtype='float32')
        init_masks = np.zeros((batch_size_eval, size, size, 1), dtype='uint8')
        fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)

        # Load postprocessing parameters
        Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(CV))
        Params_post_mat = Optimization_Info['Params'][0]
        minArea = Params_post_mat['minArea'][0][0,0]
        avgArea = Params_post_mat['avgArea'][0][0,0]
        thresh_COM0 = Params_post_mat['thresh_COM0'][0][0,0]
        thresh_COM = Params_post_mat['thresh_COM'][0][0,0]
        thresh_IOU = Params_post_mat['thresh_IOU'][0][0,0]
        thresh_consume = Params_post_mat['thresh_consume'][0][0,0]
        cons = Params_post_mat['cons'][0][0,0]
        thresh_mask = Params_post_mat['thresh_mask'][0][0,0]
        thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
        Params_post={'minArea': Params_post_mat['minArea'][0][0,0], 
            'avgArea': Params_post_mat['avgArea'][0][0,0],
            'thresh_pmap': Params_post_mat['thresh_pmap'][0][0,0], 
            'win_avg':Params_post_mat['win_avg'][0][0,0], # Params_post_mat['thresh_pmap'][0][0,0]+1)/256
            'thresh_mask': Params_post_mat['thresh_mask'][0][0,0], 
            'thresh_COM0': Params_post_mat['thresh_COM0'][0][0,0], 
            'thresh_COM': Params_post_mat['thresh_COM'][0][0,0], 
            'thresh_IOU': Params_post_mat['thresh_IOU'][0][0,0], 
            'thresh_consume': Params_post_mat['thresh_consume'][0][0,0], 
            'cons':Params_post_mat['cons'][0][0,0]}
        # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1)/256 # for published version


        # Spatial filtering preparation
        if useSF==True:
            rows1 = cv2.getOptimalDFTSize(Lx)
            cols1 = cv2.getOptimalDFTSize(Ly)
            
            try:
                Length_data=str((rows1, cols1))
                cc2 = functions_init.load_wisdom_txt('wisdom\\'+Length_data)
            except:
                cc2 = None
            
            try:
                Length_data=str((frames_init, rows1, cols1))
                cc3 = functions_init.load_wisdom_txt('wisdom\\'+Length_data)
                pyfftw.import_wisdom(cc3)
            except:
                cc3 = None

            mask2 = functions_init.plan_mask2(dims, (rows1, cols1), gauss_filt_size)
            (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft3(frames_init, (rows1, cols1))
        else:
            (mask2, bf, fft_object_b, fft_object_c) = (None, None, None, None)
            bb=np.zeros((frames_init, rowspad, colspad), dtype='float32')

        # Temporal filtering preparation
        frames_initf = frames_init - leng_tf + 1
        if useTF==True:
            video_input_init = np.ones((frames_initf, rowspad, colspad), dtype='float32')
            past_frames = np.ones((leng_past, rowspad, colspad), dtype='float32')
        else:
            (video_input_init, past_frames) = (None, None)
        
        # Pre-allocate memory for some future variables
        med_frame2 = np.ones((rowspad, colspad, 2), dtype='float32')
        video_input = np.ones((frames_initf, rowspad, colspad), dtype='float32')        
        pmaps_b_init = np.ones((frames_initf, Lx, Ly), dtype='uint8')        
        frame_input = np.ones(dimspad, dtype='float32')
        pmaps_b = np.ones(dims, dtype='uint8')
        # segs_all = None

        time_init = time.time()
        print('Parameter initialization time: {} s'.format(time_init-start))

        # %% Load data and preparation
        h5_img = h5py.File(dir_video+Exp_ID+'.h5', 'r')
        video_raw = np.array(h5_img['mov'])
        h5_img.close()
        nframes = video_raw.shape[0]
        nframesf = nframes - leng_tf + 1
        bb[:, :Lx, :Ly] = video_raw[:frames_init]
        time_load = time.time()
        print('Load data: {} s'.format(time_load-time_init))
        # time_pre = np.zeros(nframes-frames_init)
        # time_CNN = np.zeros(nframes-frames_init)
        # time_frame = np.zeros(nframes-frames_init)

        # %% Initialization using the first 30 s
        print('Initialization of algorithms using the first {} frames'.format(frames_init))
        start_init = time.time()
        med_frame3, segs_all, recent_frames = functions_init.init_online(
            bb, dims, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
            med_frame2, mask2, bf, fft_object_b, fft_object_c, Poisson_filt, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, useWT=useWT, batch_size_init=batch_size_init, p=p)
        if useTF==True:
            past_frames[:leng_tf] = recent_frames
        tuple_temp = functions_online.merge_complete(segs_all[:frames_initf], dims, Params_post)
        if show_intermediate:
            Masks_2 = functions_online.select_cons(tuple_temp)

        end_init = time.time()
        time_init = end_init-start_init
        time_frame_init = time_init/(frames_initf)*1000
        print('Initialization time: {:6f} s, {:6f} ms/frame'.format(time_init, time_frame_init))

        start_online = time.time()
        # Spatial filtering preparation for online processing. 
        # Attention: this part counts to the total time
        if useSF:
            if cc2:
                pyfftw.import_wisdom(cc2)
            (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft2((rows1, cols1))
        else:
            (bf, fft_object_b, fft_object_c) = (None, None, None)
            bb=np.zeros(dimspad, dtype='float32')

        print('Start frame by frame processing')
        # Online process for the following frames
        current_frame = leng_tf
        t_merge = frames_initf
        for t in range(frames_initf,nframesf):
            start_frame = time.time()
            # frame_raw = video_raw[ind]
            bb[:Lx, :Ly] = video_raw[t]

            # Preprocessing a frame
            # frame_input = functions_online.preprocess_online(bb, dimspad, med_frame3, frame_input, \
            #     past_frames[current_frame-leng_tf:current_frame], mask2, bf, fft_object_b, fft_object_c, \
            #     Poisson_filt, useSF=useSF, useTF=useTF, useSNR=useSNR)
            if useSF:
                par_online.fastlog(bb)
                fft_object_b()
                par_online.fastmask(bf, mask2)
                fft_object_c()
                par_online.fastexp(bb)

            # %% Temporal filtering
            if useTF:
                past_frames[current_frame] = bb[:rowspad, :colspad]
                par_online.fastconv(past_frames[current_frame-leng_tf:current_frame], frame_input, Poisson_filt)
            else:
                frame_input = bb[:rowspad, :colspad]

            # %% Median computation and normalization
            if useSNR:
                par_online.fastnormf(frame_input, med_frame3)
            else:
                par_online.fastnormback(frame_input, 0, med_frame3[0,:,:].mean())

            # frame_prob = functions_online.CNN_online(frame_input, fff, dims)
            frame_input_exp = frame_input[np.newaxis,:,:,np.newaxis] # 
            frame_prob = fff.predict(frame_input_exp, batch_size=1)
            frame_prob = frame_prob.squeeze()[:dims[0], :dims[1]] # 

            # segs = functions_online.postprocess_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT=useWT)
            par_online.fastthreshold(frame_prob, pmaps_b, thresh_pmap_float)
            segs = separateNeuron_b(pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)
            segs_all.append(segs)

            if ((t + 1 - t_merge) == merge_every) or (t==nframesf-1): # merge
                uniques, times_uniques = uniqueNeurons1_simp(segs_all[t_merge:(t+1)], thresh_COM0) # minArea,

            if ((t + 0 - t_merge) == merge_every) or (t==nframesf-1): # merge
                if uniques.size:
                    groupedneurons, times_groupedneurons = \
                        group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques)

            if ((t - 1 - t_merge) == merge_every) or (t==nframesf-1): # merge
                if uniques.size:
                    piecedneurons_1, times_piecedneurons_1 = \
                        piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)

            if ((t - 2 - t_merge) == merge_every) or (t==nframesf-1): # merge
                if uniques.size:
                    piecedneurons, times_piecedneurons = \
                        piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
                    masks_add = piecedneurons
                    times_add = [np.unique(x) + t_merge for x in times_piecedneurons]
                        
                    # %% Refine neurons using consecutive occurence
                    # masks_2_float = refine_seperate_nommin_float(masks_final_2, times_final, cons, thresh_mask)
                    # masks_2_float = masks_final_2
                    if masks_add.size:
                        masks_add = [x for x in masks_add]
                        Masksb_add = [(x >= x.max() * thresh_mask).astype('float') for x in masks_add]
                        area_add = np.array([x.nnz for x in Masksb_add]) # Masks_2.sum(axis=1).A.squeeze()
                        # if select_cons:
                        have_cons_add = functions_online.refine_seperate_cons(times_add, cons)
                        # else:
                            # have_cons=np.zeros(masks_final_2.size, dtype='bool')
                    else:
                        Masksb_add = []
                        area_add = np.array([])
                        have_cons_add = np.array([])
                else:
                    Masksb_add = []
                    masks_add = [] # masks_2_float = 
                    times_add = times_uniques
                    area_add = np.array([])
                    have_cons_add = np.array([])
                tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)
                # tuple_add = functions_online.merge_complete(segs_all[t_merge:(t+1)], dims, Params_post)
                # (Masksb_add, masks_add, times_add, area_add, have_cons_add) = tuple_add
                # times_add = [x + t_merge for x in times_add]
                # tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)

            if ((t - 3 - t_merge) == merge_every) or (t==nframesf-1): # merge
                # tuple_temp = merge_2_Jaccard(tuple_temp, tuple_add, dims, Params)
                tuple_temp = functions_online.merge_2(tuple_temp, tuple_add, dims, Params_post)
                t_merge = t+1
                if show_intermediate:
                    Masks_2 = functions_online.select_cons(tuple_temp)

            current_frame +=1
            if current_frame >= leng_past:
                current_frame = leng_tf
                past_frames[:leng_tf] = past_frames[-leng_tf:]
            end_frame = time.time()
            list_time_per[t] = end_frame - start_frame
            if t % 1000 == 0:
                print('{} frames has been processed'.format(t))

        if not show_intermediate:
            Masks_2 = functions_online.select_cons(tuple_temp)
        end_online = time.time()
        time_online = end_online-start_online
        time_frame_online = time_online/(nframesf-frames_initf)*1000
        print('Online time: {:6f} s, {:6f} ms/frame'.format(time_online, time_frame_online))

        if len(Masks_2):
            Masks_2 = sparse.vstack(Masks_2)
        else:
            Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))
        # Masks_2 = functions_online.merge_complete(segs_all, dims, Params_post)
        end_final = time.time()

        # %% Evaluation
        filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
        savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks_2':Masks_2, 'list_time_per':list_time_per[frames_initf:nframesf]})

        # %% Save information
        time_all = end_final-start_init
        time_frame_all = time_all/nframes*1000
        print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
        list_Recall[CV] = Recall
        list_Precision[CV] = Precision
        list_F1[CV] = F1
        list_time[CV] = np.array([time_init, time_online, time_all])
        list_time_frame[CV] = np.array([time_frame_init, time_frame_online, time_frame_all])

        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1,
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(dir_output+'Output_Info_All.mat', Info_dict)

    p.close()


# %%
if __name__ == '__main__':
    main()