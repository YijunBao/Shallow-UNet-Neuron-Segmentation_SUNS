import cv2
import math
import numpy as np
import time
from scipy import signal
from scipy import special
import multiprocessing as mp
import pyfftw
import h5py
import os
from scipy.io import loadmat
import sys

from suns.PreProcessing.par1 import fastexp, fastmask, fastlog, \
    fastconv, fastquant, fastnormf, fastnormback, fastmediansubtract
from suns.PreProcessing.preprocessing_functions import load_wisdom_txt, \
    plan_fft, plan_mask2, spatial_filtering, temporal_filtering, find_dataset


def median_std(result, med_frame2):
    '''Calculate median and median_based_standard_deviation from the selected video.

    Inputs: 
        result (numpy.ndarray of float32): the input video. 
            The temporal dimension is transposed to the first dimension.
        med_frame2 (3D empty numpy.ndarray of float32): 
            empty array to store the median and median-based standard deviation.

    Outputs:
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
    '''
    fastquant(result, np.array([0.5, 0.25], dtype='float32'), med_frame2)
    # med_frame2[:, :, 0] stores the median
    
    # Noise is estimated using median-based standard deviation calculated from 
    # the difference bwtween 0.5 quantile and 0.25 quantile
    temp_noise = (med_frame2[:, :, 0]-med_frame2[:, :, 1])/(math.sqrt(2)*special.erfinv(0.5))
    zeronoise = (temp_noise==0)
    # if the calculated temp_noise is 0 at some pixels, replace the median-based standard deviation 
    # with conventional stantard deviation
    if np.any(zeronoise):
        [x0, y0] = zeronoise.nonzero()
        for (x,y) in zip(x0, y0):
            new_noise = np.std(result[:, x, y])
            if new_noise>0:
                temp_noise[x, y] = new_noise
            else:
                temp_noise[x, y] = np.inf

    # med_frame2[:, :, 1] stores the median-based standard deviation
    med_frame2[:, :, 1] = np.reciprocal(temp_noise).astype('float32')
    med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing
    return med_frame3


def SNR_normalization(network_input, med_frame2=None, dims=None, frames_initf=10**4, \
        update_baseline=False, frames_init=10**6, display=True):
    '''Normalize the video to be an SNR video.
        network_input(t,x,y) = (network_input(t,x,y) - median(x,y)) / (median_based_standard_deviation(x,y)).

    Inputs: 
        network_input (numpy.ndarray of float32): the input video. 
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        dims (tuple of int, shape = (2,), default to None): lateral dimension of the video
        frames_initf (int, default to a very large number): Median and median-based standard deviation are 
            calculate from the first "frames_initf" frames of the video
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        frames_init (int, default to a very large number): Median and median-based standard deviation are 
            updated every "frames_init" frames of the video. Only used when update_baseline = True
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        No explicit output.
        In addition, "network_input" is changed to the SNR video during the function
    '''
    if display:
        start = time.time()
    if dims:
        (rows, cols) = dims
    else:
        (_, rows, cols) = network_input.shape
    if med_frame2 is None:
        med_frame2 = np.zeros(rows, cols, 2, dtype='float32')

    result = np.copy(network_input[:frames_initf, :rows, :cols].transpose([1, 2, 0]))
    # Calculate the median and medina-based standard deviation
    med_frame3 = median_std(result, med_frame2) 
    if not update_baseline:
        if display:
            endmedtime = time.time()
            print('median computation: {} s'.format(endmedtime - start))

        fastnormf(network_input[:, :rows, :cols], med_frame3)
        if display:
            endnormtime = time.time()
            print('normalization: {} s'.format(endnormtime - endmedtime))
    else:
        fastnormf(network_input[:frames_initf, :rows, :cols], med_frame3)
        for t_update in range(frames_initf, network_input.shape[0], frames_init):
            result = np.copy(network_input[t_update:t_update+frames_init, :rows, :cols].transpose([1, 2, 0]))
            fastnormf(network_input[t_update:t_update+frames_init, :rows, :cols], med_frame3)
            med_frame3 = median_std(result, med_frame2)
        if display:
            endnormtime = time.time()
            print('median computation and normalization: {} s'.format(endnormtime - start))
    # return med_frame3


def median_normalization(network_input, med_frame2=None, dims=None, frames_initf=10**4, \
        update_baseline=False, frames_init=10**6, display=True):
    '''Normalize the video by dividing to its temporal median.
        network_input(t,x,y) = network_input(t,x,y) / mean(median(x,y)).

    Inputs: 
        network_input (numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video. 
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median.
        dims (tuple of int, shape = (2,), default to None): lateral dimension of the video
        frames_initf (int, default to a very large number): Median and median-based standard deviation are 
            calculate from the first "frames_initf" frames of the video
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        frames_init (int, default to a very large number): Median and median-based standard deviation are 
            updated every "frames_init" frames of the video. Only used when update_baseline = True
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        No explicit output.
        In addition, "network_input" is changed to become the normalized video during the function
    '''
    if display:
        start = time.time()
    if dims:
        (rows, cols) = dims
    else:
        (_, rows, cols) = network_input.shape
    if med_frame2 is None:
        med_frame2 = np.zeros(rows, cols, 2, dtype='float32')

    result = np.copy(network_input[:frames_initf, :rows, :cols].transpose([1, 2, 0]))
    fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
    # med_frame2[:, :, 0] stores the median
    if not update_baseline:
        if display:
            endmedtime = time.time()
            print('median computation: {} s'.format(endmedtime - start))

        fastnormback(network_input[:, :rows, :cols], max(1, med_frame2[:,:,0].mean()))
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing
        if display:
            endnormtime = time.time()
            print('normalization: {} s'.format(endnormtime - endmedtime))
    else:
        fastnormback(network_input[:frames_initf+frames_init, :rows, :cols], max(1, med_frame2[:,:,0].mean()))
        for t_update in range(frames_initf + frames_init, network_input.shape[0], frames_init):
            result = np.copy(network_input[t_update-frames_init:t_update, :rows, :cols].transpose([1, 2, 0]))
            fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
            fastnormback(network_input[t_update:t_update+frames_init, :rows, :cols], max(1, med_frame2[:,:,0].mean()))
        if display:
            endnormtime = time.time()
            print('median computation and normalization: {} s'.format(endnormtime - start))
    # return med_frame3
    

def preprocess_complete_online(bb, dimspad, network_input=None, med_frame2=None, Poisson_filt=np.array([1]), mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, useSF=False, useTF=True, useSNR=True, \
        med_subtract=False, update_baseline=False, frames_initf=10**4, frames_init=10**6, prealloc=True, display=False):
    '''Pre-process the input video "bb" into an SNR video "network_input".
        It applies spatial filter, temporal filter, and SNR normalization in sequance. 
        Each step is optional.

    Inputs: 
        bb (3D numpy.ndarray of float32, shape = (T0,Lx0f,Ly0f)): array storing the raw video.
        dimspad (tuplel of int, shape = (2,)): lateral dimension of the padded images.
        network_input (3D empty numpy.ndarray of float32, shape = (T,Lx,Ly)): empty array to store the SNR video.
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        bf (empty, default to None): array to store the complex spectrum for FFT.
        fft_object_b (default to None): Object for forward FFT.
        fft_object_c (default to None): Object for inverse FFT.
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        frames_initf (int, default to a very large number): Median and median-based standard deviation are 
            calculate from the first "frames_initf" frames of the video
        frames_init (int, default to a very large number): Median and median-based standard deviation are 
            updated every "frames_init" frames of the video. Only used when update_baseline = True
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (3D numpy.ndarray of float32): the SNR video.
    '''
    (rowspad, colspad) = dimspad
    
    if useSF: # Homomorphic spatial filtering based on FFT
        spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2, display=display)
    # if not prealloc:
    #     del bf

    if useTF: # Temporal filtering
        temporal_filtering(bb[:, :rowspad, :colspad], network_input, Poisson_filt, display=display)
        # use copy() to enhance the stability when the lateral size of bb is 512.
        # temporal_filtering(bb[:, :rowspad, :colspad].copy(), network_input, Poisson_filt, display=display)
    else:
        network_input = bb[:, :rowspad, :colspad]

    if med_subtract and not useSF: # Subtract every frame with its median.
        if display:
            start = time.time()
        temp = np.zeros(network_input.shape[:2], dtype = 'float32')
        fastmediansubtract(network_input, temp, 2)
        if display:
            endmedsubtr = time.time()
            print('median subtraction: {} s'.format(endmedsubtr-start))

    # Median computation and normalization
    if useSNR:
        SNR_normalization(network_input, med_frame2, (rowspad, colspad), frames_initf, \
            update_baseline, frames_init, display=display)
    else:
        median_normalization(network_input, med_frame2, (rowspad, colspad), frames_initf, \
            update_baseline, frames_init, display=display)

    return network_input


def preprocess_video_online(dir_video:str, Exp_ID:str, Params:dict, frames_init=10**6, 
        dir_network_input:str = None, useSF=False, useTF=True, useSNR=True, \
        med_subtract=False, update_baseline=False, prealloc=True, display=True):
    '''Pre-process the registered video "Exp_ID" in "dir_video" into an SNR video "network_input".
        The process includes spatial filter, temporal filter, and SNR normalization. Each step is optional.
        The function does some preparations, including pre-allocating memory space (optional), FFT planing, 
        and setting filter kernels, before loading the video. 

    Inputs: 
        dir_video (str): The folder containing the input video.
            Each file must be a ".h5" file.
            The video dataset can have any name, but cannot be under any group.
        Exp_ID (str): The filer name of the input video. 
        Params (dict): Parameters for pre-processing.
            Params['gauss_filt_size'] (float): The standard deviation of the spatial Gaussian filter in pixels
            Params['Poisson_filt'] (1D numpy.ndarray of float32): The temporal filter kernel
            Params['num_median_approx'] (int): Number of frames used to compute 
                the median and median-based standard deviation
            Params['nn'] (int): Number of frames at the beginning of the video to be processed.
                The remaining video is not considered a part of the input video.
        frames_init (int, default to a very large number): Median and median-based standard deviation are 
            calculate from the first "frames_init" frames (before temporal filtering) of the video
        dir_network_input (str, default to None): The folder to save the SNR video (network_input) in hard drive.
            If dir_network_input == None, then the SNR video is not stored in hard drive
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing. 
            The shape is (T,Lx,Ly), where T is shorter than T0 due to temporal filtering, 
            and Lx and Ly are Lx0 and Ly0 padded to multiples of 8, so that the images can be process by the shallow U-Net.
        start (float): the starting time of the pipline (after the data is loaded into memory)
        In addition, the SNR video is saved in "dir_network_input" as a "(Exp_ID).h5" file containing a dataset "network_input".
    '''
    nn = Params['nn']
    if useSF:
        gauss_filt_size = Params['gauss_filt_size']
    Poisson_filt = Params['Poisson_filt']
    # num_median_approx = Params['num_median_approx']

    h5_video = os.path.join(dir_video, Exp_ID + '.h5')
    h5_file = h5py.File(h5_video,'r')
    dset = find_dataset(h5_file)
    (nframes, rows, cols) = h5_file[dset].shape
    # Make the lateral number of pixels a multiple of 8, so that the CNN can process them 
    rowspad = math.ceil(rows/8)*8 
    colspad = math.ceil(cols/8)*8
    # Only keep the first "nn" frames to process
    nframes = min(nframes,nn)
    
    if useSF:
        # lateral dimensions slightly larger than the raw video but faster for FFT
        rows1 = cv2.getOptimalDFTSize(rows)
        cols1 = cv2.getOptimalDFTSize(cols)

        if display:
            print(rows, cols, nn, '->', rows1, cols1, nn)
            start_plan = time.time()
        # if the learned wisdom files have been saved, load them. Otherwise, learn wisdom later
        Length_data=str((nn, rows1, cols1))
        cc = load_wisdom_txt(os.path.join('wisdom', Length_data))
        if cc:
            pyfftw.import_wisdom(cc)

        # FFT planning
        bb, bf, fft_object_b, fft_object_c = plan_fft(nn, (rows1, cols1), prealloc)
        if display:
            end_plan = time.time()
            print('FFT planning: {} s'.format(end_plan - start_plan))

        # %% Initialization: Calculate the spatial filter and set variables.
        mask2 = plan_mask2((rows, cols), (rows1, cols1), gauss_filt_size)
    else:
        bb=np.zeros((nframes, rowspad, colspad), dtype='float32')
        fft_object_b = None
        fft_object_c = None
        bf = None
        end_plan = time.time()
        mask2 = None

    # %% Initialization: Set variables.
    if useTF:
        leng_tf = Poisson_filt.size
        nframesf = nframes - leng_tf + 1 # Number of frames after temporal filtering
    else:
        nframesf = nframes
    frames_initf = frames_init - leng_tf + 1
    if prealloc:
        network_input = np.ones((nframesf, rowspad, colspad), dtype='float32')
        med_frame2 = np.ones((rowspad, colspad, 2), dtype='float32')
    else:
        network_input = np.zeros((nframesf, rowspad, colspad), dtype='float32')
        med_frame2 = np.zeros((rowspad, colspad, 2), dtype='float32')
    # median_decimate = max(1,nframes//num_median_approx)
    if display:
        end_init = time.time()
        print('Initialization: {} s'.format(end_init - end_plan))
    
    # %% Load the raw video into "bb"
    for t in range(nframes): # use this one to save memory
        bb[t, :rows, :cols] = np.array(h5_file[dset][t])
    h5_file.close()
    if display:
        end_load = time.time()
        print('data loading: {} s'.format(end_load - end_init))

    start = time.time() # The pipline starts after the video is loaded into memory
    network_input = preprocess_complete_online(bb, (rowspad,colspad), network_input, med_frame2, \
        Poisson_filt, mask2, bf, fft_object_b, fft_object_c, useSF, useTF, useSNR, \
        med_subtract, update_baseline, frames_initf, frames_init, prealloc, display)

    if display:
        end = time.time()
        print('total per frame: {} ms'.format((end - start) / nframes *1000))

    # if "dir_network_input" is not None, save network_input to an ".h5" file
    if dir_network_input:
        f = h5py.File(os.path.join(dir_network_input, Exp_ID+".h5"), "w")
        f.create_dataset("network_input", data = network_input)
        f.close()
        if display:
            end_saving = time.time()
            print('Network_input saving: {} s'.format(end_saving - end))
        
    return network_input, start

