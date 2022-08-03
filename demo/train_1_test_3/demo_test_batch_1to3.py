# %%
import os
import numpy as np
import time
import h5py
import sys

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '../..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2
from suns.run_suns import suns_batch

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
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
    rate_hz = 10 # frame rate of the video
    Mag = 6/8 # spatial magnification compared to ABO videos (0.785 um/pixel). # Mag = 0.785 / pixel_size

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    filename_TF_template = '../YST_spike_tempolate.h5' # File name storing the temporal filter kernel
    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    h5f.close()
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    Poisson_filt = Poisson_filt/Poisson_filt.sum()
    # # Alternative temporal filter kernel using a single exponential decay function
    # decay = 0.8 # decay time constant (unit: second)
    # leng_tf = np.ceil(rate_hz*decay)+1
    # Poisson_filt = np.exp(-np.arange(leng_tf)/rate_hz/decay)
    # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')

    # %% Set processing options
    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
        # Can only be used when spatial filtering is not used. 
    prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
    batch_size_eval = 200 # batch size in CNN inference
    useWT=False # True if using additional watershed
    display=True # True if display information about running time 
    #-------------- End user-defined parameters --------------#

    dir_parent = os.path.join(dir_video, 'noSF 1to3') # folder to save all the processed data
    dir_output = os.path.join(dir_parent, 'output_masks') # folder to save the segmented masks and the performance scores
    dir_params = os.path.join(dir_parent, 'output_masks') # folder of the optimized hyper-parameters
    weights_path = os.path.join(dir_parent, 'Weights') # folder of the trained CNN
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
        
    # dictionary of pre-processing parameters
    Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'Poisson_filt': Poisson_filt}

    p = mp.Pool()
    nvideo = len(list_Exp_ID)
    list_CV = list(range(0,nvideo))
    num_CV = len(list_CV)
    # arrays to save the recall, precision, F1, total processing time, and average processing time per frame
    list_Recall = np.zeros((num_CV, nvideo, 1))
    list_Precision = np.zeros((num_CV, nvideo, 1))
    list_F1 = np.zeros((num_CV, nvideo, 1))
    list_time = np.zeros((num_CV, nvideo, 4))
    list_time_frame = np.zeros((num_CV, nvideo, 4))


    for CV in list_CV:
        filename_CNN = os.path.join(weights_path, 'Model_CV{}.h5'.format(CV)) # The path of the CNN model.

        # load optimal post-processing parameters
        Optimization_Info = loadmat(os.path.join(dir_params, 'Optimization_Info_{}.mat'.format(CV)))
        Params_post_mat = Optimization_Info['Params'][0]
        # dictionary of all optimized post-processing parameters.
        Params_post={
            # minimum area of a neuron (unit: pixels).
            'minArea': Params_post_mat['minArea'][0][0,0], 
            # average area of a typical neuron (unit: pixels) 
            'avgArea': Params_post_mat['avgArea'][0][0,0],
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
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

        for (eid, Exp_ID) in enumerate(list_Exp_ID):
            if eid == CV:
                continue
            print('CV ', CV, ', Video ', Exp_ID)

            # The entire process of SUNS batch
            Masks, Masks_2, times_active, time_total, time_frame = suns_batch(
                dir_video, Exp_ID, filename_CNN, Params_pre, Params_post, batch_size_eval, \
                useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, \
                useWT=useWT, prealloc=prealloc, display=display, p=p)
            savemat(os.path.join(dir_output, 'Output_Masks_CV{}_{}.mat'.format(CV, Exp_ID)), \
                {'Masks':Masks, 'times_active':times_active}, do_compression=True)

            # %% Evaluation of the segmentation accuracy compared to manual ground truth
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            data_GT=loadmat(filename_GT)
            GTMasks_2 = data_GT['GTMasks_2'].transpose()
            (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
            print({'Recall':Recall, 'Precision':Precision, 'F1':F1})

            # %% Save recall, precision, F1, total processing time, and average processing time per frame
            list_Recall[CV, eid] = Recall
            list_Precision[CV, eid] = Precision
            list_F1[CV, eid] = F1
            list_time[CV, eid] = time_total
            list_time_frame[CV, eid] = time_frame

        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(os.path.join(dir_output, 'Output_Info_All.mat'), Info_dict)

    p.close()


