# %%
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import par1
import h5py
import os

from preprocessing_functions import process_video, generate_masks


# %%
if __name__ == '__main__':
    list_exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    dir_video = 'D:\\ABO\\20 percent\\'
    dir_save = dir_video + 'ShallowUNet\\complete\\'        
    dir_network_input = dir_save+"network_input\\"
    dir_GTMasks = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_'
    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 

    nn = 23200 # nframes # cv2.getOptimalDFTSize(nframes)
    gauss_filt_size = 50  # signa in pixels
    num_median_approx = 1000
    network_baseline = 0 # 64.0
    network_SNRscale = 1 # 32.0
    list_thred_ratio = [6, 7] # range(4,6) # 
    useSF=True
    useTF=True
    useSNR=True
    # frame_rate = 30
    # decay = 0.2
    # leng_tf = math.ceil(frame_rate*decay) # 6
    # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
    # Poisson_filt = Poisson_filt / Poisson_filt.sum()
    h5f = h5py.File('GCaMP6f_spike_tempolate_mean.h5','r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze()[3:12].astype('float32')
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'network_baseline':network_baseline, 'network_SNRscale':network_SNRscale, 
        'nn':nn, 'Poisson_filt': Poisson_filt}
        # 'frame_rate':frame_rate, 'decay':decay, 'leng_tf':leng_tf, 

    for Exp_ID in list_exp_ID: #[-1:]
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