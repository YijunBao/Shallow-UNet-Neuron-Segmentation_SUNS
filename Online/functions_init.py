# %%
import os
# import cv2
import math
import numpy as np
# import matplotlib.pyplot as plt
import time
import h5py
import sys
import pyfftw
from scipy import signal
from scipy import special

# import random
# import tensorflow as tf
from scipy.io import savemat, loadmat
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import par_online
import par1
from par3 import fastthreshold
# from preprocessing_functions import process_video, process_video_prealloc
# from complete_post import complete_segment
from seperate_multi import separateNeuron_b


def load_wisdom_txt(dir_wisdom):
    try:
        file3 = open(dir_wisdom+'x1.txt', 'rb')
        cc = file3.read()
        file3.close()
        file3 = open(dir_wisdom+'x2.txt', 'rb')
        dd = file3.read()
        file3.close()
        file3 = open(dir_wisdom+'x3.txt', 'rb')
        ee = file3.readline()
        cc = (cc, dd, ee)
    except:
        cc = None
    return cc

def plan_fft3(frames_init, dims1):
    (rows1, cols1) = dims1 # frame_raw.shape
    bb = pyfftw.zeros_aligned((frames_init, rows1, cols1), dtype='float32', n=8)
    bf = pyfftw.zeros_aligned((frames_init, rows1, cols1//2+1), dtype='complex64', n=8)
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
    return bb, bf, fft_object_b, fft_object_c

def plan_fft2(dims1):
    (rows1, cols1) = dims1 # frame_raw.shape
    bb = pyfftw.zeros_aligned((rows1, cols1), dtype='float32', n=8)
    bf = pyfftw.empty_aligned((rows1, cols1//2+1), dtype='complex64', n=8)
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
    return bb, bf, fft_object_b, fft_object_c

def plan_mask2(dims, dims1, gauss_filt_size):
    (rows, cols) = dims
    (rows1, cols1) = dims1 # frame_raw.shape
    gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
    mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
    mask1_col = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
    mask2 = np.outer(mask1_row, mask1_col)
    mask = np.fft.fftshift(mask2)
    # mask2 = np.zeros((rows1, cols1//2+1), dtype='float32')
    mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
    return mask2


def init_online(bb, dims, video_input, pmaps_b, fff, thresh_pmap_float, Params_post, med_frame2=None, mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, Poisson_filt=np.array([1]), \
        useSF=True, useTF=True, useSNR=True, useWT=False, batch_size_init=1, p=None):

    [Lx, Ly]=dims
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)
    # video_input, start = process_video_prealloc(dir_video, Exp_ID, Params_pre, useSF=useSF, useTF=useTF, useSNR=useSNR)
    video_input, med_frame3 = preprocess_init(bb, dimspad, video_input, med_frame2, Poisson_filt, mask2, \
        bf, fft_object_b, fft_object_c, useSF=useSF, useTF=useTF, useSNR=useSNR)
    if useTF==True:
        recent_frames = video_input[-(Poisson_filt.size):]
    else:
        recent_frames = None

    # %% Network inference
    video_input = np.expand_dims(video_input, axis=-1)
    prob_map = fff.predict(video_input, batch_size=batch_size_init)

    prob_map = prob_map.squeeze()[:, :Lx, :Ly]
    # # pmaps =(prob_map*256-0.5).astype(np.uint8)
    # pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)
    # fastuint(prob_map, pmaps)
    # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
    # fastcopy(prob_map, pmaps) 
    # pmaps = prob_map

    # %% PostProcessing
    # Masks_2 = complete_segment(pmaps_b, Params_post, p=p, useWT=useWT)
    segs_init = segment_init(pmaps_b, Params_post, p=p, useWT=useWT)

    return med_frame3, segs_init, recent_frames


def preprocess_init(bb, dimspad, video_input=None, med_frame2=None, Poisson_filt=np.array([1]), mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, useSF=False, useTF=True, useSNR=True):

    (rowspad, colspad) = dimspad
    # %% Homomorphic spatial filtering
    if useSF:
        par1.fastlog(bb)
        fft_object_b()
        par1.fastmask(bf, mask2)
        fft_object_c()
        par1.fastexp(bb)

    # %% Temporal filtering
    if useTF:
        par1.fastconv(bb[:, :rowspad, :colspad], video_input, Poisson_filt)
    else:
        video_input = bb[:, :rowspad, :colspad]

    # %% Median computation and normalization
    result = np.copy(video_input.transpose([1, 2, 0]))
    if useSNR:
        par1.fastquant(result, np.array([0.5, 0.25], dtype='float32'), med_frame2)
        temp_noise = (med_frame2[:, :, 0]-med_frame2[:, :, 1])/(math.sqrt(2)*special.erfinv(0.5))
        # np.clip(temp_noise, 0.5, None, out=temp_noise)
        # temp_noise = np.sqrt(med_frame2[:, :, 0])
        zeronoise = (temp_noise==0)
        if np.any(zeronoise):
            [x0, y0] = zeronoise.nonzero()
            for (x,y) in zip(x0, y0):
                new_noise = np.std(video_input[:, x, y])
                if new_noise>0:
                    temp_noise[x, y] = new_noise
                else:
                    temp_noise[x, y] = np.inf

        med_frame2[:, :, 1] = np.reciprocal(temp_noise).astype('float32')
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1]))
        med_frame3 = med_frame2.transpose([2, 0, 1])
        par1.fastnormf(video_input, med_frame3)
    else:
        par1.fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
        par1.fastnormback(video_input, 0, med_frame2[:,:,0].mean())
        med_frame3 = med_frame2.transpose([2, 0, 1])

    return video_input, med_frame3


def segment_init(pmaps: np.ndarray, Params: dict, useMP=True, useWT=False, p=None):
    '''The input probablity map must be pre-thresholded'''
    # nframes = len(pmaps)
    # (Lx, Ly) = pmaps[0].shape
    minArea = Params['minArea']
    avgArea = Params['avgArea']
    thresh_pmap = Params['thresh_pmap']
    # thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    # thresh_COM = Params['thresh_COM']
    # thresh_IOU = Params['thresh_IOU']
    # thresh_consume = Params['thresh_consume']
    # cons = Params['cons']
    # win_avg = Params['win_avg']
    # if useMP: # %% Run segmentation with multiprocessing
    #     p = mp.Pool(mp.cpu_count())

    if useMP: # %% Run segmentation with multiprocessing
        segs = p.starmap(separateNeuron_b, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
    else: # %% Run segmentation without multiprocessing
        segs = []
        nframes = pmaps.shape[0]
        for ind in range(nframes):
            segs.append(separateNeuron_b(pmaps[ind], thresh_pmap, minArea, avgArea, useWT)) #, eng
    # num_neurons = sum([x[1].size for x in segs])

    # if num_neurons==0:
    #     print('No masks found. Please lower minArea or thresh_pmap.')
    #     Masks_2 = sparse.csc_matrix((0,Lx*Ly), dtype='bool')
    # else:
    #     uniques, times_uniques = uniqueNeurons1_simp(segs, thresh_COM0) # minArea, , p
    #     groupedneurons, times_groupedneurons = \
    #         group_neurons(uniques, thresh_COM, thresh_mask, (dims[1], dims[2]), times_uniques)
    #     piecedneurons_1, times_piecedneurons_1 = \
    #         piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
    #     piecedneurons, times_piecedneurons = \
    #         piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
    #     # %% Final result
    #     masks_final_2 = piecedneurons
    #     times_final = [np.unique(x) for x in times_piecedneurons]
            
    #     # %% Refine neurons using consecutive occurence
    #     Masks_2 = refine_seperate_nommin(masks_final_2, times_final, cons, thresh_mask)

    return segs#, Masks_2 # masks_final_2, times_final #, matname_output

