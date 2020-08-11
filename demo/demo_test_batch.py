# %%
import os
import math
import numpy as np
import time
import h5py
import sys

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

import par1
from preprocessing_functions import preprocess_video
from par3 import fastthreshold
from shallow_unet import get_shallow_unet
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import complete_segment

# %%
if __name__ == '__main__':
    # %% setting parameters
    Dimens = (120,88) # lateral dimensions of the video
    nframes = 3000 # number of frames for each video
    Mag = 6/8 # spatial magnification compared to ABO videos.

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
    batch_size_eval = 200 # batch size in CNN inference
    useWT=False # True if using additional watershed

    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = 'data\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 

    dir_parent = dir_video + 'complete\\' # folder to save all the processed data
    dir_sub = ''
    dir_output = dir_parent + dir_sub + 'output_masks\\' # folder to save the segmented masks and the performance scores
    dir_params = dir_parent + dir_sub + 'output_masks\\' # folder of the optimized hyper-parameters
    weights_path = dir_parent + dir_sub + 'Weights\\' # folder of the trained CNN
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 

    # %% pre-processing parameters
    nn = nframes
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    (Lx, Ly) = Dimens # lateral dimensions of the video
    h5f = h5py.File('YST_spike_tempolate.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
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
    list_time = np.zeros((num_CV, 4))
    list_time_frame = np.zeros((num_CV, 4))


    for CV in list_CV:
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)

        start = time.time()
        # load CNN model
        fff = get_shallow_unet()
        fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))
        # run CNN inference once to warm up
        init_imgs = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='float32')
        init_masks = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='uint8')
        fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)
        del init_imgs, init_masks
        time_init = time.time()

        # load optimal post-processing parameters
        Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(CV))
        Params_post_mat = Optimization_Info['Params'][0]
        # dictionary of all optimized post-processing parameters.
        Params_post={
            # minimum area of a neuron (unit: pixels).
            'minArea': Params_post_mat['minArea'][0][0,0], 
            # average area of a typical neuron (unit: pixels) 
            'avgArea': Params_post_mat['avgArea'][0][0,0],
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1.5)
            'thresh_pmap': Params_post_mat['thresh_pmap'][0][0,0], 
            # values higher than "thresh_mask" times the maximum value of the mask are set to one.
            'thresh_mask': Params_post_mat['thresh_mask'][0][0,0], 
            # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
            'thresh_COM0': Params_post_mat['thresh_COM0'][0][0,0], 
            # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
            'thresh_COM': Params_post_mat['thresh_COM'][0][0,0], 
            # minimum IoU of two masks to be considered the same neuron
            'thresh_IOU': Params_post_mat['thresh_IOU'][0][0,0], 
            # minimum consume ratio of two masks to be considered the same neuron
            'thresh_consume': Params_post_mat['thresh_consume'][0][0,0], 
            # minimum consecutive number of frames of active neurons
            'cons':Params_post_mat['cons'][0][0,0]}
        # convert the uint8 probability threshold back to float
        thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
        # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1)/256 # for published version
        print('Initialization time: {} s'.format(time_init-start))


        # %% pre-processing including loading data
        # start = time.time()
        video_input, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc)
        end_pre = time.time()
        nframes = video_input.shape[0]
        time_pre = end_pre-start
        time_frame_pre = time_pre/nframes*1000
        print('Pre-Processing time: {:6f} s, {:6f} ms/frame'.format(time_pre, time_frame_pre))

        # %% CNN inference
        video_input = np.expand_dims(video_input, axis=-1)
        start_network = time.time()
        prob_map = fff.predict(video_input, batch_size=batch_size_eval)
        end_network = time.time()
        time_CNN = end_network-start_network
        time_frame_CNN = time_CNN/nframes*1000
        print('CNN Infrence time: {:6f} s, {:6f} ms/frame'.format(time_CNN, time_frame_CNN))

        # %% post-processing
        prob_map = prob_map.squeeze()[:, :Lx, :Ly]
        print(Params_post)
        Params_post['thresh_pmap'] = None # Avoid repeated thresholding in postprocessing
        start_post = time.time()
        pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
        # threshold the probability map to binary activity
        fastthreshold(prob_map, pmaps_b, thresh_pmap_float)

        # the rest of post-processing. The result is a 2D sparse matrix of the segmented neurons
        Masks_2 = complete_segment(pmaps_b, Params_post, display=True, p=p, useWT=useWT)
        finish = time.time()
        time_post = finish-start_post
        time_frame_post = time_post/nframes*1000
        print('Post-Processing time: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

        # %% Evaluation of the segmentation accuracy compared to manual ground truth
        filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
        # convert to a 3D array of the segmented neurons
        Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly))
        savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks':Masks})

        # %% Save recall, precision, F1, total processing time, and average processing time per frame
        time_all = finish-start
        time_frame_all = time_all/nframes*1000
        print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
        list_Recall[CV] = Recall
        list_Precision[CV] = Precision
        list_F1[CV] = F1
        list_time[CV] = np.array([time_pre, time_CNN, time_post, time_all])
        list_time_frame[CV] = np.array([time_frame_pre, time_frame_CNN, time_frame_post, time_frame_all])

        del video_input, pmaps_b, prob_map, fff
        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(dir_output+'Output_Info_All.mat', Info_dict)

    p.close()


