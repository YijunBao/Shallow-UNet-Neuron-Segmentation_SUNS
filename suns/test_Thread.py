# %%
import os
import sys
import math
import numpy as np
import time
import h5py
import pyfftw
from scipy import sparse
from scipy import signal
from scipy import special
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat, loadmat
import multiprocessing as mp
import subprocess
import threading

sys.path.insert(1, '..') # the path containing "suns" folder
sys.path.insert(1, '..\\..') # the path containing "suns" folder
# from suns.Online.functions_online import merge_2, merge_2_nocons, merge_complete, select_cons, \
#     preprocess_online, CNN_online, separate_neuron_online, refine_seperate_cons_online
# from suns.Online.functions_init import init_online, plan_fft2
from suns.PreProcessing.preprocessing_functions import preprocess_video, \
    plan_fft, plan_mask2, load_wisdom_txt, SNR_normalization, median_normalization
# from suns.Online.preprocessing_functions_online import preprocess_video_online
# from suns.PostProcessing.par3 import fastthreshold
# from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
#     group_neurons, piece_neurons_IOU, piece_neurons_consume
# from suns.PostProcessing.complete_post import complete_segment
from suns.Network.shallow_unet import get_shallow_unet


def wait(pause):
    print('start')
    time.sleep(pause)
    print('finish')


def test_process(a_in, a_out):
    a_out[:] = a_in[:]
    print(a_in)
    print(a_out[:])
    return a_out

# %%
if __name__ == '__main__':
    # dims = (256,256)
    # mask2 = np.zeros(dims, 'float32')
    # p = mp.Process(target=plan_mask2_process, args=mask2)
    # p = mp.Process(target = wait, args = (2,))
    le = 6
    a_in = np.arange(le)
    a_out = np.zeros(le,'uint16')
    # a_out = mp.Array('H', 6)
    # p = mp.Process(target = test_process, args = (a_in, a_out))
    p = threading.Thread(target = test_process, args = (a_in, a_out))
    print(a_in)
    print(a_out[:])
    print(p.is_alive())
    p.start()
    print(p.is_alive())
    p.join()
    print(p.is_alive())
    print(a_in)
    print(a_out[:])
    # time.sleep(2)
    # print(p.is_alive())
    # time.sleep(2)
    # print(p.is_alive())
    # time.sleep(2)
    # print(p.is_alive())
    print('finish all')

