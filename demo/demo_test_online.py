# %%
import os
import cv2
import math
import numpy as np
import time
import h5py
import sys
import pyfftw
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(0, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
sys.path.insert(1, '..\\online')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from unet4_best import get_unet
from evaluate_post import GetPerformance_Jaccard_2
import functions_online
import functions_init
import par_online
from seperate_multi import separateNeuron_b
from combine import uniqueNeurons1_simp, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume


# %%
if __name__ == '__main__':
    # %% setting parameters
    rate_hz = 10 # frame rate of the video
    Dimens = (120,88) # lateral dimensions of the video
    nframes = 3000 # number of frames for each video
    Mag = 6/8 # spatial magnification compared to ABO videos.

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
    useWT=False # True if using additional watershed
    show_intermediate=True # True if screen neurons with consecutive frame requirement after every merge

    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = 'data\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 

    merge_every = rate_hz # number of frames every merge
    frames_init = 30 * rate_hz # number of frames used for initialization
    batch_size_init = 100 # batch size in CNN inference during initalization

    dir_parent = dir_video + 'complete\\' # folder to save all the processed data
    dir_sub = ''
    dir_output = dir_parent + dir_sub + 'output_masks online\\' # folder to save the segmented masks and the performance scores
    dir_params = dir_parent + dir_sub + 'output_masks\\' # folder of the optimized hyper-parameters
    weights_path = dir_parent + dir_sub + 'Weights\\' # folder of the trained CNN
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 

    # %% pre-processing parameters
    nn = nframes
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = frames_init # number of frames used to caluclate median and median-based standard deviation
    (Lx, Ly) = Dimens # lateral dimensions of the video
    dims = (Lx,Ly) # lateral dimensions of the video
    # zero-pad the lateral dimensions to multiples of 8, suitable for CNN
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)

    h5f = h5py.File('YST_spike_tempolate.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    leng_tf = Poisson_filt.size
    leng_past = 2*leng_tf # number of past frames stored for temporal filtering
    # dictionary of pre-processing parameters
    Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    p = mp.Pool()
    list_CV = list(range(0,4))
    num_CV = len(list_CV)
    # arrays to save the recall, precision, F1, total processing time, and average processing time per frame
    list_Recall = np.zeros((num_CV, 1))
    list_Precision = np.zeros((num_CV, 1))
    list_F1 = np.zeros((num_CV, 1))
    list_time = np.zeros((num_CV, 3))
    list_time_frame = np.zeros((num_CV, 3))


    for CV in list_CV:
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)
        list_time_per = np.zeros(nn)

        start = time.time()
        # load CNN model
        fff = get_unet()
        fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))
        # run CNN inference once to warm up
        init_imgs = np.zeros((batch_size_init, Lx, Ly, 1), dtype='float32')
        init_masks = np.zeros((batch_size_init, Lx, Ly, 1), dtype='uint8')
        fff.evaluate(init_imgs, init_masks, batch_size=batch_size_init)
        del init_imgs, init_masks

        # load optimal post-processing parameters
        time_init = time.time()
        Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(CV))
        Params_post_mat = Optimization_Info['Params'][0]
        # minimum area of a neuron (unit: pixels).
        minArea = Params_post_mat['minArea'][0][0,0]
        # average area of a typical neuron (unit: pixels) 
        avgArea = Params_post_mat['avgArea'][0][0,0]
        # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
        thresh_COM0 = Params_post_mat['thresh_COM0'][0][0,0]
        # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
        thresh_COM = Params_post_mat['thresh_COM'][0][0,0]
        # minimum IoU of two masks to be considered the same neuron
        thresh_IOU = Params_post_mat['thresh_IOU'][0][0,0]
        # minimum consume ratio of two masks to be considered the same neuron
        thresh_consume = Params_post_mat['thresh_consume'][0][0,0]
        # minimum consecutive number of frames of active neurons
        cons = Params_post_mat['cons'][0][0,0]
        # values higher than "thresh_mask" times the maximum value of the mask are set to one.
        thresh_mask = Params_post_mat['thresh_mask'][0][0,0]
        # threshould of probablity map (float)
        thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
        # dictionary of all optimized post-processing parameters.
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
        # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
        # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1)/256 # for published version
        print('Initialization time: {} s'.format(time_init-start))


        # Spatial filtering preparation
        if useSF==True:
            # lateral dimensions slightly larger than the raw video but faster for FFT
            rows1 = cv2.getOptimalDFTSize(rowspad)
            cols1 = cv2.getOptimalDFTSize(colspad)
            
            # if the learned 2D and 3D wisdom files have been saved, load them. Otherwise, learn wisdom later
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

            # mask for spatial filter
            mask2 = functions_init.plan_mask2(dims, (rows1, cols1), gauss_filt_size)
            # FFT planning
            (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft3(frames_init, (rows1, cols1))
        else:
            (mask2, bf, fft_object_b, fft_object_c) = (None, None, None, None)
            bb=np.zeros((frames_init, rowspad, colspad), dtype='float32')

        # Temporal filtering preparation
        frames_initf = frames_init - leng_tf + 1
        if useTF==True:
            if prealloc:
                # past frames stored for temporal filtering
                past_frames = np.ones((leng_past, rowspad, colspad), dtype='float32')
            else:
                past_frames = np.zeros((leng_past, rowspad, colspad), dtype='float32')
        else:
            past_frames = None
        
        if prealloc:
            # Pre-allocate memory for some future variables
            med_frame2 = np.ones((rowspad, colspad, 2), dtype='float32')
            video_input = np.ones((frames_initf, rowspad, colspad), dtype='float32')        
            pmaps_b_init = np.ones((frames_initf, Lx, Ly), dtype='uint8')        
            frame_input = np.ones(dimspad, dtype='float32')
            pmaps_b = np.ones(dims, dtype='uint8')
        else:
            med_frame2 = np.zeros((rowspad, colspad, 2), dtype='float32')
            video_input = np.zeros((frames_initf, rowspad, colspad), dtype='float32')        
            pmaps_b_init = np.zeros((frames_initf, Lx, Ly), dtype='uint8')        
            frame_input = np.zeros(dimspad, dtype='float32')
            pmaps_b = np.zeros(dims, dtype='uint8')

        time_init = time.time()
        print('Parameter initialization time: {} s'.format(time_init-start))


        # %% Load raw video
        h5_img = h5py.File(dir_video+Exp_ID+'.h5', 'r')
        video_raw = np.array(h5_img['mov'])
        h5_img.close()
        nframes = video_raw.shape[0]
        nframesf = nframes - leng_tf + 1
        bb[:, :Lx, :Ly] = video_raw[:frames_init] #, :Lx, :Ly
        time_load = time.time()
        print('Load data: {} s'.format(time_load-time_init))


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
        # %% Online processing for the following frames
        current_frame = leng_tf
        t_merge = frames_initf
        for t in range(frames_initf,nframesf):
            start_frame = time.time()
            bb[:Lx, :Ly] = video_raw[t]

            if useSF: # Spatial filtering
                par_online.fastlog(bb)
                fft_object_b()
                par_online.fastmask(bf, mask2)
                fft_object_c()
                par_online.fastexp(bb)

            if useTF: # Temporal filtering
                past_frames[current_frame] = bb[:rowspad, :colspad]
                par_online.fastconv(past_frames[current_frame-leng_tf:current_frame], frame_input, Poisson_filt)
            else:
                frame_input = bb[:rowspad, :colspad]

            if useSNR: # SNR normalization
                par_online.fastnormf(frame_input, med_frame3)
            else:
                par_online.fastnormback(frame_input, 0, med_frame3[0,:,:].mean())

            # CNN inference
            frame_input_exp = frame_input[np.newaxis,:,:,np.newaxis]
            frame_prob = fff.predict(frame_input_exp, batch_size=1)
            frame_prob = frame_prob.squeeze()[:dims[0], :dims[1]]

            # post-processing
            # threshold the probability map to binary activity
            par_online.fastthreshold(frame_prob, pmaps_b, thresh_pmap_float)
            # spatial clustering each frame to form connected regions representing active neurons
            segs = separateNeuron_b(pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)
            segs_all.append(segs)

            # temporal merging 1: combine neurons with COM distance smaller than thresh_COM0
            if ((t + 1 - t_merge) == merge_every) or (t==nframesf-1):
                uniques, times_uniques = uniqueNeurons1_simp(segs_all[t_merge:(t+1)], thresh_COM0) # minArea,

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
                        have_cons_add = functions_online.refine_seperate_cons(times_add, cons)
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
                tuple_temp = functions_online.merge_2(tuple_temp, tuple_add, dims, Params_post)
                t_merge = t+1
                if show_intermediate:
                    Masks_2 = functions_online.select_cons(tuple_temp)

            current_frame +=1
            # Update the stored latest frames when it runs out: move them "leng_tf" ahead
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

        # final result. Masks_2 is a 2D sparse matrix of the segmented neurons
        if len(Masks_2):
            Masks_2 = sparse.vstack(Masks_2)
        else:
            Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))
        end_final = time.time()

        # %% Evaluation of the segmentation accuracy compared to manual ground truth
        filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
        # convert to a 3D array of the segmented neurons
        Masks = np.reshape(Masks_2.todense().A, (Masks_2.shape[0], Lx, Ly))
        savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks':Masks})

        # %% Save recall, precision, F1, total processing time, and average processing time per frame
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
        del video_input, fff

    p.close()


