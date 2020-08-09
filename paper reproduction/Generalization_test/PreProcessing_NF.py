# %%
import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import os

sys.path.insert(0, '..\\PreProcessing')
from preprocessing_functions import generate_masks
import par1
from functions_other_data import process_video_prealloc_others, data_info_neurofinder, temporal_filter


# %%
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder = list_neurofinder[0:4]
    # dataset_type = 'test' # 'train' # 'test' # 
    for dataset_type in {'train', 'test'}: # 'train' 
        if dataset_type == 'train':
            list_Exp_ID = list_neurofinder # [0:1] # list_caiman_Exp_ID[3:4] # 
        elif dataset_type == 'test':
            list_Exp_ID = [Exp_ID+'.test' for Exp_ID in list_neurofinder] # [0:1] # list_caiman_Exp_ID[3:4] # 

        dir_video = 'E:\\NeuroFinder\\{} videos\\'.format(dataset_type)
        dir_parent = dir_video + 'ShallowUNet\\noSF\\'
        dir_save = dir_parent
        dir_network_input = dir_parent + 'network_input\\'
        dir_GTMasks = 'C:\\Matlab Files\\STNeuroNet-master\\Markings\\Neurofinder\\{}\\Grader1\\FinalMasks_'.format(dataset_type)
        if not os.path.exists(dir_network_input):
            os.makedirs(dir_network_input) 

        nn = 10000 # nframes # cv2.getOptimalDFTSize(nframes)
        gauss_filt_size = 50  # signa in pixels
        num_median_approx = 1000
        network_baseline = 0 # 64.0
        network_SNRscale = 1 # 32.0
        list_thred_ratio = [3] # range(4,8)
        useSF=False # False
        useTF=True
        useSNR=True

        for Exp_ID in list_Exp_ID: #[1:]
            # %% Process video
            print('Video ', Exp_ID)
            fname_info = dir_video + 'neurofinder.' + Exp_ID + '\\info.json'
            info, Mxy = data_info_neurofinder(fname_info)
            gauss_filt_size = gauss_filt_size*Mxy  # signa in pixels
            # dims_raw = info['dimensions'][::-1]

            # frame_rate = 30
            # decay = 0.2
            # leng_tf = math.ceil(frame_rate*decay) # 6
            # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
            # Poisson_filt = Poisson_filt / Poisson_filt.sum()
            # h5f = h5py.File('GCaMP6f_spike_tempolate_mean.h5','r')
            # Poisson_filt = np.array(h5f['filter_tempolate']).squeeze()[3:12].astype('float32')
            Poisson_filt = temporal_filter(info)
            Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
                'network_baseline':network_baseline, 'network_SNRscale':network_SNRscale, 
                'nn':nn, 'Poisson_filt': Poisson_filt}
                # 'frame_rate':frame_rate, 'decay':decay, 'leng_tf':leng_tf, 

            # start = time.time()
            print(Poisson_filt.size)
            video_input, _, _ = process_video_prealloc_others(dir_video, Exp_ID, Params, dir_network_input, useSF=useSF, useTF=useTF, useSNR=useSNR) #
            # (nframesf, rows, cols) = video_input.shape
            # end_process = time.time()

            # %% Determine active frames using FISSA
            file_mask = dir_GTMasks + Exp_ID[1] + Exp_ID[3:5] + '.mat'
            generate_masks(video_input, file_mask, list_thred_ratio, dir_save, Exp_ID)
        

# %%
# if __name__ == '__main__':
#     print('+1s')