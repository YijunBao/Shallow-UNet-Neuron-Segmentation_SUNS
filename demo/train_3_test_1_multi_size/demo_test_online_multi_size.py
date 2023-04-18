# %%
import os
import numpy as np
import time
import h5py
import sys
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '../..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2
from suns.run_suns import suns_online

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config = config)
else: # tf_version == 2:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# %%
if __name__ == '__main__':
    #-------------- Start user-defined parameters --------------#
    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = '../data' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices. 'FinalMasks_' is a prefix of the file names. 
    dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_') 

    # %% set video parameters
    list_rate_hz = [10] * len(list_Exp_ID) # frame rate of all the videos. Close frame rates are preferred.
    list_Mag = [6/8] * len(list_Exp_ID) # spatial magnification compared to ABO videos (0.785 um/pixel). # Mag = 0.785 / pixel_size

    # %% Set processing options
    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
        # Can only be used when spatial filtering is not used. 
    update_baseline=True # True if the median and median-based std is updated every "frames_init" frames.
    prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
    batch_size_init = 100 # batch size in CNN inference during initalization
    useWT=False # True if using additional watershed
    show_intermediate=True # True if screen neurons with consecutive frame requirement after every merge
    display=True # True if display information about running time 
    #-------------- End user-defined parameters --------------#


    dir_parent = os.path.join(dir_video, 'noSF_multi_size') # folder to save all the processed data
    dir_output = os.path.join(dir_parent, 'output_masks online') # folder to save the segmented masks and the performance scores
    dir_params = os.path.join(dir_parent, 'output_masks') # folder of the optimized hyper-parameters
    weights_path = os.path.join(dir_parent, 'Weights') # folder of the trained CNN
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 

    p = mp.Pool()
    nvideo = len(list_Exp_ID)
    list_CV = list(range(0,nvideo))
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
        rate_hz = list_rate_hz[CV]
        Mag = list_Mag[CV]
        frames_init = 30 * rate_hz # number of frames used for initialization
        num_median_approx = frames_init # number of frames used to caluclate median and median-based standard deviation
        merge_every = rate_hz # number of frames every merge

        if useTF:
            # %% set temporal filter
            filename_TF_template = '../YST_spike_tempolate.h5' # file name of the temporal filter kernel
            h5f = h5py.File(filename_TF_template,'r')
            Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
            h5f.close()

            # Rescale the filter template according to "rate_hz"
            # It assumes the calcium sensors are the same, but the frame rates are different
            fs_template = 10 # frame rate of the filter tempolate
            peak = Poisson_filt.argmax()
            length = Poisson_filt.shape
            xp = np.arange(-peak,length-peak,1)/fs_template
            x = np.arange(np.round(-peak*rate_hz/fs_template), np.round(length-peak*rate_hz/fs_template), 1)/rate_hz
            Poisson_filt = np.interp(x,xp,Poisson_filt).astype('float32')
            
            Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
            Poisson_filt = Poisson_filt/Poisson_filt.sum()

            # # Alternative temporal filter kernel using a single exponential decay function
            # decay = 0.8 # decay time constant (unit: second)
            # leng_tf = np.ceil(rate_hz*decay)+1
            # Poisson_filt = np.exp(-np.arange(leng_tf)/rate_hz/decay)
            # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
        else:
            Poisson_filt=np.array([1], dtype='float32')

        # dictionary of pre-processing parameters, and adjust with magnification
        gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
        Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
            'Poisson_filt': Poisson_filt}

        filename_video = os.path.join(dir_video, Exp_ID+'.h5') # The path of the file of the input video.
        filename_CNN = os.path.join(weights_path, 'Model_CV{}.h5'.format(CV)) # The path of the CNN model.
        # If you used cross_validation == 'use_all' in training, you need to change the "CV" in "format(CV)"
        # to the number of tranining videos used. 

        # load optimal post-processing parameters, and adjust with magnification
        filename_params_post = os.path.join(dir_params, 'Optimization_Info_{}.mat'.format(CV))
        # If you used cross_validation == 'use_all' in training, you need to change the "CV" in "format(CV)"
        # to the number of tranining videos used. 
        Optimization_Info = loadmat(filename_params_post)
        Params_post_mat = Optimization_Info['Params'][0]
        Params_post={
            # minimum area of a neuron (unit: pixels).
            'minArea': np.round(Params_post_mat['minArea'][0][0,0] * Mag**2), 
            # average area of a typical neuron (unit: pixels) 
            'avgArea': np.round(Params_post_mat['avgArea'][0][0,0] * Mag**2),
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
            'thresh_pmap': Params_post_mat['thresh_pmap'][0][0,0], 
            # values higher than "thresh_mask" times the maximum value of the mask are set to one.
            'thresh_mask': Params_post_mat['thresh_mask'][0][0,0], 
            # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
            'thresh_COM0': np.round(Params_post_mat['thresh_COM0'][0][0,0] * Mag), 
            # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
            'thresh_COM': np.round(Params_post_mat['thresh_COM'][0][0,0] * Mag), 
            # minimum IoU of two masks to be considered the same neuron
            'thresh_IOU': Params_post_mat['thresh_IOU'][0][0,0], 
            # minimum consume ratio of two masks to be considered the same neuron
            'thresh_consume': Params_post_mat['thresh_consume'][0][0,0], 
            # minimum consecutive number of frames of active neurons
            'cons':Params_post_mat['cons'][0][0,0]}

        # The entire process of SUNS online
        Masks, Masks_2, times_active, time_total, time_frame, list_time_per = suns_online(
            filename_video, filename_CNN, Params_pre, Params_post, \
            frames_init, merge_every, batch_size_init, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, \
            update_baseline=update_baseline, useWT=useWT, \
            show_intermediate=show_intermediate, prealloc=prealloc, display=display, p=p)
        savemat(os.path.join(dir_output, 'Output_Masks_{}.mat'.format(Exp_ID)), \
            {'Masks':Masks, 'times_active':times_active, 'list_time_per':list_time_per}, do_compression=True)

        # %% Evaluation of the segmentation accuracy compared to manual ground truth
        filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose().astype('bool')
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1})

        # %% Save recall, precision, F1, total processing time, and average processing time per frame
        list_Recall[CV] = Recall
        list_Precision[CV] = Precision
        list_F1[CV] = F1
        list_time[CV] = time_total
        list_time_frame[CV] = time_frame

        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(os.path.join(dir_output, 'Output_Info_All.mat'), Info_dict)

    p.close()


