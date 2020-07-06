# %%
import os
# import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import sys
from scipy import sparse
from scipy.io import savemat, loadmat

from functions_other_data import load_caiman_roi, \
    load_caiman_video, data_info_caiman, data_info_neurofinder



# %%
if __name__ == '__main__':
    list_caiman = [ 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t',
                    'J115', 'J123', 'K53', 'YST']
    dataset_type = 'caiman' # 'neurofinder_train' # 'neurofinder_test' # 
    list_Exp_ID = list_caiman[:5]

    dir_video = 'F:\\CaImAn data\\'
    dir_output = 'F:\\CaImAn data\\WEBSITE\\' 
    size = 513
    # rows = cols = size
    Lx = Ly = size

    for vid in range(5):
        Exp_ID = list_Exp_ID[vid]
        print('Video ', Exp_ID)
        info, Mxy = data_info_caiman(dir_output + Exp_ID + '\\info.json')
        dims_raw = info['dimensions']

        video = load_caiman_video(dir_video+'images_', Exp_ID)
        f = h5py.File(dir_output + '{}_513.h5'.format(Exp_ID), "w")
        f.create_dataset("mov", data = video)
        f.close()

        # GTMasks = load_caiman_roi(dir_output, Exp_ID, dims_raw[1:])
        GTMasks = load_caiman_roi(dir_output, Exp_ID, (Lx,Ly))
        GTMasks = GTMasks.transpose((2,1,0))
        savemat(dir_output + 'FinalMasks_{}_513.mat'.format(Exp_ID), {'FinalMasks':GTMasks})
