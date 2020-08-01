# %%
import os
# import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import sys

# import random
# import tensorflow as tf
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import par1
from preprocessing_functions import process_video, process_video_prealloc
# from par2 import fastuint, fastcopy
from par3 import fastthreshold
from unet4_best import get_unet
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import complete_segment

# %%
if __name__ == '__main__':
    radius = 6
    # rate_hz = 10
    # decay_time = 0.75
    Dimens = (120,88)
    nframes = 3000
    Mag = radius/8

    thred_std = 3
    num_train_per = 2400
    BATCH_SIZE = 10
    NO_OF_EPOCHS = 50
    batch_size_eval = 50
    useSF=True
    useTF=True
    useSNR=True
    prealloc=False
    useWT=False

    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']
    dir_video = 'data\\'
    dir_GTMasks = dir_video + 'FinalMasks_'

    dir_parent = dir_video + 'complete\\'
    dir_sub = ''
    dir_output = dir_parent + dir_sub + 'output_masks\\' #
    dir_params = dir_parent + dir_sub + 'output_masks\\'
    weights_path = dir_parent + dir_sub + 'Weights\\'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 

    # %% PreProcessing parameters
    nn = nframes # 23200 # nframes # cv2.getOptimalDFTSize(nframes)
    gauss_filt_size = 50*Mag  # signa in pixels
    num_median_approx = 1000
    (Lx, Ly) = Dimens
    h5f = h5py.File('YST_spike_tempolate.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)]
    Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    # %% PostProcessing parameters
    p = mp.Pool() #mp.cpu_count()

    list_CV = list(range(0,4))
    num_CV = len(list_CV)
    list_Recall = np.zeros((num_CV, 1))
    list_Precision = np.zeros((num_CV, 1))
    list_F1 = np.zeros((num_CV, 1))
    list_time = np.zeros((num_CV, 4))
    list_time_frame = np.zeros((num_CV, 4))

    for CV in list_CV:
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)

        start = time.time()
        fff = get_unet() #size
        fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))
        init_imgs = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='float32')
        init_masks = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='uint8')
        fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)
        del init_imgs, init_masks
        time_init = time.time()
        Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(CV))
        Params_post_mat = Optimization_Info['Params'][0]
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
        thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
        # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1)/256 # for published version
        print('Initialization time: {} s'.format(time_init-start))

        # %% Load data and preparation
        # h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        # test_imgs = np.expand_dims(np.array(h5_img['network_input']), axis=-1)
        # h5_img.close()
        # nmask = 100
        # time_load = time.time()
        # print('Load data: {} s'.format(time_load-time_init))

        # %% PreProcessing including loading data
        # start = time.time()
        video_input, start = process_video(dir_video, Exp_ID, Params_pre, useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc)
        end_pre = time.time()
        nframes = video_input.shape[0]
        time_pre = end_pre-start
        time_frame_pre = time_pre/nframes*1000
        print('PreProcessing time: {:6f} s, {:6f} ms/frame'.format(time_pre, time_frame_pre))

        # %% Network inference
        video_input = np.expand_dims(video_input, axis=-1)
        start_network = time.time()
        prob_map = fff.predict(video_input, batch_size=batch_size_eval)
        end_network = time.time()
        time_CNN = end_network-start_network
        time_frame_CNN = time_CNN/nframes*1000
        print('CNN Infrence time: {:6f} s, {:6f} ms/frame'.format(time_CNN, time_frame_CNN))

        prob_map = prob_map.squeeze()[:, :Lx, :Ly]
        # %% PostProcessing
        print(Params_post)
        start_post = time.time()
        # # pmaps =(prob_map*256-0.5).astype(np.uint8)
        pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
        fastthreshold(prob_map, pmaps_b, thresh_pmap_float)
        # fastuint(prob_map, pmaps)
        # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
        # fastcopy(prob_map, pmaps) 
        # pmaps = prob_map

        Masks_2 = complete_segment(pmaps_b, Params_post, display=True, p=p, useWT=useWT)
        # Masks = np.reshape(Masks_2.todense().A, (Masks_2.shape[0], Lx, Ly))
        finish = time.time()
        time_post = finish-start_post
        time_frame_post = time_post/nframes*1000
        print('PostProcessing time: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

        # %% Evaluation
        filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
        savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks_2':Masks_2})

        # %% Save information
        time_all = finish-start
        time_frame_all = time_all/nframes*1000
        print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
        list_Recall[CV] = Recall
        list_Precision[CV] = Precision
        list_F1[CV] = F1
        list_time[CV] = np.array([time_pre, time_CNN, time_post, time_all])
        list_time_frame[CV] = np.array([time_frame_pre, time_frame_CNN, time_frame_post, time_frame_all])

        del video_input, pmaps_b, prob_map, fff#, Masks
        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(dir_output+'Output_Info_All.mat', Info_dict)

    p.close()


