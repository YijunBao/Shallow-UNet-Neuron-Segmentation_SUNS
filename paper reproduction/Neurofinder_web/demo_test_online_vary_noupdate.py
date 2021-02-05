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

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config = config)


# %%
if __name__ == '__main__':
    #-------------- Start user-defined parameters --------------#
    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID_full = [['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', \
                        '00.06', '00.07', '00.08', '00.09', '00.10', '00.11'], \
                        ['01.00', '01.01'], ['02.00', '02.01'], ['03.00'], ['04.00'], ['05.00']] 
                        # '04.01' is renamed as '05.00', because the imaging condition is different from '04.00'
    list_rate_hz = [7, 7.5, 8, 7.5, 6.75, 3]
    list_px_um = [1/1.15, 1/0.8, 1/1.15, 1.17, 0.8, 1.25]

    for ind_set in [0,1,2,3,4,5]:# [3,4]: # 
        # %% set video parameters
        list_Exp_ID = list_Exp_ID_full[ind_set]
        rate_hz = list_rate_hz[ind_set] # frame rate of the video
        Mag = 0.785*list_px_um[ind_set] # spatial magnification compared to ABO videos (0.785 um/pixel). # Mag = 0.785 / pixel_size
        # folder of the raw videos
        dir_video = 'E:\\NeuroFinder\\web\\train videos\\' + list_Exp_ID[0][:2]
        # folder of the ".mat" files stroing the GT masks in sparse 2D matrices. 'FinalMasks_' is a prefix of the file names. 
        dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_') 

        # %% set pre-processing parameters
        gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
        frames_init = 300 # number of frames used for initialization
        num_median_approx = frames_init # number of frames used to caluclate median and median-based standard deviation
        filename_TF_template = list_Exp_ID[0][:2]+'_spike_tempolate.h5' # File name storing the temporal filter kernel
        h5f = h5py.File(filename_TF_template,'r')
        Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
        Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
        Poisson_filt = Poisson_filt/Poisson_filt.sum()
        # # Alternative temporal filter kernel using a single exponential decay function
        # decay = 0.8 # decay time constant (unit: second)
        # leng_tf = np.ceil(rate_hz*decay)+1
        # Poisson_filt = np.exp(-np.arange(leng_tf)/rate_hz/decay)
        # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')

        # %% Set processing options
        useSF=bool(int(sys.argv[1])) # True if spatial filtering is used in pre-processing.
        useTF=bool(int(sys.argv[2])) # True if temporal filtering is used in pre-processing.
        useSNR=bool(int(sys.argv[3])) # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
        med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
            # Can only be used when spatial filtering is not used. 
        update_baseline=False # True if the median and median-based std is updated every "frames_init" frames.
        prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
                # Achieve faster speed at the cost of higher memory occupation.
        batch_size_init = 100 # batch size in CNN inference during initalization
        useWT=False # True if using additional watershed
        show_intermediate=True # True if screen neurons with consecutive frame requirement after every merge
        display=True # True if display information about running time 
        merge_every = rate_hz # number of frames every merge
        #-------------- End user-defined parameters --------------#


        dir_parent = os.path.join(dir_video, sys.argv[4]) # folder to save all the processed data
        dir_output = os.path.join(dir_parent, 'output_masks online no_update') # folder to save the segmented masks and the performance scores
        dir_params = os.path.join(dir_parent, 'output_masks') # folder of the optimized hyper-parameters
        weights_path = os.path.join(dir_parent, 'Weights') # folder of the trained CNN
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 

        # dictionary of pre-processing parameters
        if not useTF:
            Poisson_filt=np.array([1])
        Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
            'Poisson_filt': Poisson_filt}

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
            filename_video = os.path.join(dir_video, Exp_ID+'.h5') # The path of the file of the input video.
            filename_CNN = os.path.join(weights_path, 'Model_CV{}.h5'.format(len(list_Exp_ID))) # The path of the CNN model.
            # Load post-processing hyper-parameters
            filename_params_post = os.path.join(dir_params, 'Optimization_Info_{}.mat'.format(len(list_Exp_ID)))
            Optimization_Info = loadmat(filename_params_post)
            Params_post_mat = Optimization_Info['Params'][0]
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

            # The entire process of SUNS online
            Masks, Masks_2, times_active, time_total, time_frame, list_time_per = suns_online(
                filename_video, filename_CNN, Params_pre, Params_post, \
                frames_init, merge_every, batch_size_init, \
                useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, \
                update_baseline=update_baseline, useWT=useWT, \
                show_intermediate=show_intermediate, prealloc=prealloc, display=display, p=p)

            # %% Evaluation of the segmentation accuracy compared to manual ground truth
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            data_GT=loadmat(filename_GT)
            GTMasks_2 = data_GT['GTMasks_2'].transpose().astype('bool')
            (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
            print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
            savemat(os.path.join(dir_output, 'Output_Masks_{}.mat'.format(Exp_ID)), \
                {'Masks':Masks, 'times_active':times_active, 'list_time_per':list_time_per}, do_compression=True)

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


