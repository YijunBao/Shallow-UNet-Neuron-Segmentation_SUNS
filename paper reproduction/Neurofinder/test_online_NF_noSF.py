# %%
import os
import numpy as np
import time
import h5py
import sys
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2
from suns.run_suns import suns_online


# %%
if __name__ == '__main__':
    # %% setting parameters
    list_neurofinder_train = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    list_neurofinder_test = [x+'.test' for x in list_neurofinder_train]
    px_um = [1/0.8, 1/0.8, 1/1.15, 1/1.15, 0.8, 1.25]
    list_Mag = [x*0.78 for x in px_um]
    list_rate_hz = [7.5, 7.5, 8, 8, 6.75, 3] # [3] * 6 # 
    Dimens = [(504,504), (504,504), (464,504), (464,504), (416,480), (416,480)] # lateral dimension of the video
    list_nframes_train = [2250, 1825, 8000, 8000, 3000, 3000]
    list_nframes_test = [2250, 5000, 8000, 8000, 3000, 3000]

    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=True # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
    useWT=False # True if using additional watershed
    show_intermediate=True # True if screen neurons with consecutive frame requirement after every merge
    display=True # True if display information about running time 
    p = mp.Pool()

    for trainset_type in {'train', 'test'}: # 
        testset_type = list({'train','test'}-{trainset_type})[0]
        # %% set folders
        if trainset_type == 'train':
            list_nframes = list_nframes_test
            list_Exp_ID_train = list_neurofinder_train
            list_Exp_ID = list_neurofinder_test
        else: # if trainset_type == 'test':
            list_nframes = list_nframes_train
            list_Exp_ID_train = list_neurofinder_test
            list_Exp_ID = list_neurofinder_train

        # file names of the ".h5" files storing the raw videos. 
        # folder of the raw videos
        dir_video = 'E:\\NeuroFinder\\{} videos\\'.format(testset_type)
        dir_video_train = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
        dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 

        dir_parent = dir_video + 'noSF\\' # folder to save all the processed data
        dir_parent_train = dir_video_train + 'noSF\\' # folder to save all the processed data
        dir_output = dir_parent + 'output_masks online\\' # folder to save the segmented masks and the performance scores
        dir_params = dir_parent_train + 'output_masks\\' # folder of the optimized hyper-parameters
        weights_path = dir_parent_train + 'Weights\\' # folder of the trained CNN
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 

        nvideo = len(list_Exp_ID)
        # list_CV = list(range(0,nvideo))
        # num_CV = len(list_CV)
        # arrays to save the recall, precision, F1, total processing time, and average processing time per frame
        list_Recall = np.zeros((nvideo, 1))
        list_Precision = np.zeros((nvideo, 1))
        list_F1 = np.zeros((nvideo, 1))
        list_time = np.zeros((nvideo, 3))
        list_time_frame = np.zeros((nvideo, 3))

        for (ind_video, Exp_ID) in enumerate(list_Exp_ID): # 
            rate_hz = list_rate_hz[ind_video] # frame rate of the video
            nframes = list_nframes[ind_video] # number of frames for each video
            Mag = list_Mag[ind_video] # spatial magnification compared to ABO videos.
            # thred_std = list_thred_std[ind_video] # SNR threshold used to determine when neurons are active
            dims = (rows, cols) = Dimens[ind_video] # size of the network input and output
            (Lx, Ly) = (rows, cols) # size of the original video
            num_total = nframes # number of frames of the video

            # %% pre-processing parameters
            nn = nframes
            gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
            num_median_approx = frames_init # number of frames used to caluclate median and median-based standard deviation
            # dims = (Lx, Ly) = Dimens # lateral dimensions of the video
            filename_TF_template = 'GCaMP6s_spike_tempolate_mean.h5' # file name of the temporal filter kernel

            merge_every = rate_hz # number of frames every merge
            frames_init = 30 * rate_hz # number of frames used for initialization
            batch_size_init = 100 # batch size in CNN inference during initalization

            if useTF:
                h5f = h5py.File(filename_TF_template,'r')
                Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
                Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
            else:
                Poisson_filt=np.array([1])

            # dictionary of pre-processing parameters
            Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
                'nn':nn, 'Poisson_filt': Poisson_filt}


            # for CV in list_CV:
            Exp_ID_train = list_Exp_ID_train[ind_video]
            print('Video ', Exp_ID)
            filename_video = dir_video+Exp_ID+'.h5' # The path of the file of the input video.
            filename_CNN = weights_path+'Model_{}.h5'.format(Exp_ID_train) # The path of the CNN model.

            # Load post-processing hyper-parameters
            filename_params_post = dir_params+'Optimization_Info_{}.mat'.format(Exp_ID_train)
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
            Masks, Masks_2, time_total, time_frame = suns_online(
                filename_video, filename_CNN, Params_pre, Params_post, \
                dims, frames_init, merge_every, batch_size_init, \
                useSF=useSF, useTF=useTF, useSNR=useSNR, useWT=useWT, \
                show_intermediate=show_intermediate, prealloc=prealloc, display=display, p=p)

            # %% Evaluation of the segmentation accuracy compared to manual ground truth
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            data_GT=loadmat(filename_GT)
            GTMasks_2 = data_GT['GTMasks_2'].transpose()
            (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
            print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
            savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks':Masks})

            # %% Save recall, precision, F1, total processing time, and average processing time per frame
            list_Recall[ind_video] = Recall
            list_Precision[ind_video] = Precision
            list_F1[ind_video] = F1
            list_time[ind_video] = time_total
            list_time_frame[ind_video] = time_frame

        Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
            'list_time':list_time, 'list_time_frame':list_time_frame}
        savemat(dir_output+'Output_Info_All.mat', Info_dict)

    p.close()


