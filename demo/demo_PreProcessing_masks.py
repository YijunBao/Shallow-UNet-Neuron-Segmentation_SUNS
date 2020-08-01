# %%
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import os

sys.path.insert(1, '..\\PreProcessing')
from preprocessing_functions import process_video, generate_masks
import par1


# %%
if __name__ == '__main__':
    radius = 6
    # rate_hz = 10
    # decay_time = 0.75
    # Dimens = (120,88)
    nframes = 3000
    Mag = radius/8

    useSF=True
    useTF=True
    useSNR=True
    prealloc=False

    list_Exp_ID = ['YST_part11_int', 'YST_part12_int', 'YST_part21_int', 'YST_part22_int']
    dir_video = 'data\\'
    dir_GTMasks = dir_video + 'FinalMasks_'

    dir_save = dir_video + 'complete\\'        
    dir_network_input = dir_save+"network_input\\"
    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 

    nn = nframes # 23200 # nframes # cv2.getOptimalDFTSize(nframes)
    gauss_filt_size = 50*Mag  # signa in pixels
    num_median_approx = 1000
    list_thred_ratio = [3] # 
    h5f = h5py.File('YST_spike_tempolate.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)]
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    for Exp_ID in list_Exp_ID: #
        # %% Process video
        video_input, _ = process_video(dir_video, Exp_ID, Params, dir_network_input, useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #

        # %% Determine active frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat'
        generate_masks(video_input, file_mask, list_thred_ratio, dir_save, Exp_ID)
        del video_input
        
