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
    list_name_video = ['J115', 'J123', 'K53', 'YST']
    list_radius = [8,10,8,6] # 
    list_rate_hz = [30,30,30,10] # 
    list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(224,224),(216,152), (248,248),(120,88)]
    list_nframes = [90000, 41000, 116043, 3000]
    ID_part = ['_part11', '_part12', '_part21', '_part22']
    list_Mag = [x/8 for x in list_radius]

    useSF=True
    useTF=True
    useSNR=True

    for ind_video in range(0,4): # [3]: # 
        name_video = list_name_video[ind_video]
        list_Exp_ID = [name_video+x for x in ID_part]
        dir_video = 'F:\\CaImAn data\\WEBSITE\\divided_data\\'+name_video+'\\'
        dir_GTMasks = dir_video + 'FinalMasks_'

        dir_save = dir_video + 'ShallowUNet\\complete\\'        
        dir_network_input = dir_save+"network_input\\"
        if not os.path.exists(dir_network_input):
            os.makedirs(dir_network_input) 

        nn = list_nframes[ind_video] # 23200 # nframes # cv2.getOptimalDFTSize(nframes)
        gauss_filt_size = 50*list_Mag[ind_video]  # signa in pixels
        num_median_approx = 1000
        network_baseline = 0 # 64.0
        network_SNRscale = 1 # 32.0
        list_thred_ratio = list(range(4,9)) # [6, 7] # 
        # frame_rate = 30
        # decay = 0.2
        # leng_tf = math.ceil(frame_rate*decay) # 6
        # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
        # Poisson_filt = Poisson_filt / Poisson_filt.sum()
        h5f = h5py.File('C:\\Matlab Files\\Generalization_test\\{}_spike_tempolate.h5'.format(name_video),'r')
        Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
        Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)]
        Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
            'network_baseline':network_baseline, 'network_SNRscale':network_SNRscale, 
            'nn':nn, 'Poisson_filt': Poisson_filt}
            # 'frame_rate':frame_rate, 'decay':decay, 'leng_tf':leng_tf, 

        for Exp_ID in list_Exp_ID[3:]: #
            # %% Process video
            # start = time.time()
            video_input, _ = process_video(dir_video, Exp_ID, Params, dir_network_input, useSF=useSF, useTF=useTF, useSNR=useSNR) #
            # (nframesf, rows, cols) = video_input.shape
            # end_process = time.time()

            # %% Determine active frames using FISSA
            file_mask = dir_GTMasks + Exp_ID + '.mat'
            generate_masks(video_input, file_mask, list_thred_ratio, dir_save, Exp_ID)
            del video_input
            

    # %%
    # if __name__ == '__main__':
    #     print('+1s')