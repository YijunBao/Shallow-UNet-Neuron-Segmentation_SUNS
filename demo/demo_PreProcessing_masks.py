# %%
import sys
import cv2
import numpy as np
import time
import h5py
import os

sys.path.insert(1, '..\\PreProcessing')
from preprocessing_functions import preprocess_video, generate_masks
import par1


# %%
if __name__ == '__main__':
    # %% setting parameters
    nframes = 3000 # number of frames for each video
    Mag = 6/8 # spatial magnification compared to ABO videos.

    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.

    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = 'data\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 

    dir_save = dir_video + 'complete\\' # folder to save all the processed data
    dir_network_input = dir_save+"network_input\\" # folder to save the SNR videos
    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 

    nn = nframes
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = [3] # A list of SNR threshold used to determine when neurons are active.
    h5f = h5py.File('YST_spike_tempolate.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}


    # pre-processing for training
    for Exp_ID in list_Exp_ID: #
        # %% Pre-process video
        video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #

        # %% Determine active neurons in all frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks(video_input, file_mask, list_thred_ratio, dir_save, Exp_ID)
        del video_input
        
