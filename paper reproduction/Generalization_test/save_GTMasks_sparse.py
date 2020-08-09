# %%
import os
import sys
# import io
import numpy as np
# import cv2
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import time
import h5py
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

sys.path.insert(0, '..\\PreProcessing')
from functions_other_data import load_caiman_roi2, load_neurofinder_roi2, \
    data_info_caiman, data_info_neurofinder
    
# %% 
if __name__ == '__main__':
    #time.sleep(3*3600)
    # %% 
    list_CV = list(range(0,6))
    # Lx = 487
    # Ly = 487
    useWT = False
    thred_std = 5
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    # useSF=False
    # useTF=True
    # useSNR=True
    list_caiman = [ 'N.00.00', 'N.01.00', 'N.02.00', 'N.03.00.t', 'N.04.00.t',
                    'J115', 'J123', 'K53', 'YST']
    list_neurofinder_train = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder_test = ['01.00.test', '01.01.test', '02.00.test', '02.01.test', '04.00.test', '04.01.test']
    list_neurofinder_test = [Exp_ID+'.test' for Exp_ID in list_neurofinder_train]
    dataset_type = 'neurofinder_test' # 'neurofinder_train' # 'neurofinder_test' # 
    
    if dataset_type == 'caiman':
        list_Exp_ID = list_caiman # [0:1] # list_caiman_Exp_ID[3:4] # 
    elif dataset_type == 'neurofinder_train':
        list_Exp_ID = list_neurofinder_train # [0:1] # list_caiman_Exp_ID[3:4] # 
    elif dataset_type == 'neurofinder_test':
        list_Exp_ID = list_neurofinder_test # [0:1] # list_caiman_Exp_ID[3:4] # 

    if dataset_type == 'caiman':
        dir_video = 'E:\\CaImAn data\\images_'
        dir_output = 'E:\\CaImAn data\\from ABO 275\\' #
        dir_GTMasks = 'E:\\CaImAn data\\WEBSITE\\' 
    elif dataset_type == 'neurofinder_train':
        dir_video = 'E:\\NeuroFinder\\train videos\\'
        dir_output = 'E:\\NeuroFinder\\train videos\\ShallowUNet\\' #
        dir_GTMasks = 'C:\\Matlab Files\\STNeuroNet-master\\Markings\\Neurofinder\\train\\Grader1\\FinalMasks_' 
    elif dataset_type == 'neurofinder_test':
        dir_video = 'E:\\NeuroFinder\\test videos\\'
        dir_output = 'E:\\NeuroFinder\\test videos\\ShallowUNet\\' #
        dir_GTMasks = 'C:\\Matlab Files\\STNeuroNet-master\\Markings\\Neurofinder\\test\\Grader1\\FinalMasks_' 
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    dir_pmap = dir_output
    dir_temp = dir_output
    
    # %%
    for CV in list_CV: # list(range(0,6))+list(range(7,10)): #
        Exp_ID = list_Exp_ID[CV]    
        print('Video ', Exp_ID)
        if dataset_type == 'caiman':
            fname_info = dir_video + Exp_ID + '\\info.json'
            info, Mxy = data_info_caiman(fname_info)
            dims_raw = info['dimensions']
        elif 'neurofinder' in dataset_type:
            fname_info = dir_video + 'neurofinder.' + Exp_ID + '\\info.json'
            info, Mxy = data_info_neurofinder(fname_info)
            dims_raw = info['dimensions'][::-1]

        if dataset_type == 'caiman':
            GTMasks_2 = load_caiman_roi2(dir_GTMasks, Exp_ID, (dims_raw[1],dims_raw[2]))
        elif 'neurofinder' in dataset_type:
            GTMasks_2 = load_neurofinder_roi2(dir_GTMasks, Exp_ID)
        GTMasks_2_mat = GTMasks_2.transpose()

        savemat(dir_video + 'GTMasks_' + Exp_ID + '_sparse.mat', {'GTMasks_2': GTMasks_2_mat})