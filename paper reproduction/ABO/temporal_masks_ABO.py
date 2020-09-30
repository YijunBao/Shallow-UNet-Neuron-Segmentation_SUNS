# %%
import sys
import os
import random
import time
import glob
import numpy as np
import math
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\..') # the path containing "suns" folder
# os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

# from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.PreProcessing.generate_masks import generate_masks_from_traces
# from suns.train_CNN_params import train_CNN, parameter_optimization_cross_validation


# %%
if __name__ == '__main__':
    sub_folder = sys.argv[3] # e.g. 'noSF'
    # %% setting parameters
    rate_hz = 30 # frame rate of the video
    Dimens = (487,487) # lateral dimensions of the video
    nn = 23200 # number of frames used for preprocessing. 
        # Can be slightly larger than the number of frames of a video
    num_total = 23000 # number of frames used for CNN training. 
        # Can be slightly smaller than the number of frames of a video
    Mag = 1 # spatial magnification compared to ABO videos.

    thred_std = int(sys.argv[2]) # SNR threshold used to determine when neurons are active
    num_train_per = 200 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 200 # batch size in CNN inference

    useSF=bool(int(sys.argv[1])) # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.
    useWT=False # True if using additional watershed
    load_exist=False # True if using temp files already saved in the folders
    use_validation = True # True to use a validation set outside the training set
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "leave_one_out"
    Params_loss = {'DL':1, 'BCE':0, 'FL':100, 'gamma':1, 'alpha':0.25} # Parameters of the loss function

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    # folder of the raw videos
    dir_video = 'D:\\ABO\\20 percent\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 
    dir_parent = dir_video + sub_folder + '\\' # folder to save all the processed data
    dir_network_input = dir_parent + 'network_input\\' # folder of the SNR videos
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks

    # if not os.path.exists(dir_network_input):
    #     os.makedirs(dir_network_input) 
    # if not os.path.exists(weights_path):
    #     os.makedirs(weights_path) 
    # if not os.path.exists(training_output_path):
    #     os.makedirs(training_output_path) 
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output) 
    # if not os.path.exists(dir_temp):
    #     os.makedirs(dir_temp) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the original video
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = 'GCaMP6f_spike_tempolate_mean.h5'

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
        # %% Pre-process video
        # video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
        #     useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #
        # h5_img = h5py.File(dir_network_input+Exp_ID+'.h5', 'r')
        # video_input = np.array(h5_img['network_input'])

        # %% Determine active neurons in all frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)
        # del video_input
