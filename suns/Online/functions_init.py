# %%
import os
import sys
import math
import numpy as np
import time
import h5py
import pyfftw
from scipy import signal
from scipy import special

from scipy.io import savemat, loadmat
import multiprocessing as mp

from suns.PreProcessing.preprocessing_functions import preprocess_complete
from suns.PostProcessing.par3 import fastthreshold
from suns.PostProcessing.seperate_neurons import separate_neuron


def plan_fft2(dims1):
    '''Plan FFT for pyfftw for a 2D image.

    Inputs: 
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the image.

    Outputs:
        bb(2D numpy.ndarray of float32): array of the real video.
        bf(2D numpy.ndarray of complex64): array of the complex spectrum.
        fft_object_b(pyfftw.FFTW): Object for forward FFT.
        fft_object_c(pyfftw.FFTW): Object for inverse FFT.
    '''
    (rows1, cols1) = dims1
    bb = pyfftw.zeros_aligned((rows1, cols1), dtype='float32', n=8)
    # No pre-allocation, because this step is not executed in the initialization stage,
    # So the running time counts to the total time.
    bf = pyfftw.empty_aligned((rows1, cols1//2+1), dtype='complex64', n=8)
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
    return bb, bf, fft_object_b, fft_object_c


def init_online(bb, dims, dimsnb, network_input, pmaps_b, fff, thresh_pmap_float, Params_post, med_frame2=None, mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, Poisson_filt=np.array([1]), \
        useSF=True, useTF=True, useSNR=True, med_subtract=False, useMP=True, useWT=False, batch_size_init=1, p=None):
    '''Process the initial part of a video into a list of segmented masks for every frame with statistics.
        It includes complete pre-processing, CNN inference, and a half of post-processing.
        The postprocessing includes the steps for each individual frame, but does not include temporal merging.
        The outputs include "segs" containing the segmented masks for every frame with statistics, 
        and "med_frame3" and "recent_frames" used for future frame-by-frame processing.

    Inputs: 
        bb(3D numpy.ndarray of float32): array storing the raw video.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw images.
        dimsnb (tuplel of int, shape = (2,)): lateral dimension of the padded images for numba calculation in pre-processing.
        network_input (3D empty numpy.ndarray of float32): empty array to store the SNR video of the inital video.
        pmaps_b(3D empty numpy.ndarray of uint8): array to store the probablity map of the inital video.
        fff(tf.keras.Model): CNN model.
        thresh_pmap_float(float, range in 0 to 1): Threshold of probablity map.
        Params_post(dict): Parameters for post-processing.
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
        bf(3D numpy.ndarray of complex64, default to None): array to store the complex spectrum for FFT.
        fft_object_b(pyfftw.FFTW, default to None): Object for forward FFT.
        fft_object_c(pyfftw.FFTW, default to None): Object for inverse FFT.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        median_decimate (int, default to 1): Median and median-based standard deviation are 
            calculate from every "median_decimate" frames of the video
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        useWT (bool, default to False): Indicator of whether watershed is used. 
        batch_size_init (int, default to 1): Batch size for CNN inference in the initalization stage.
        p (multiprocessing.Pool, default to None): 

    Outputs:
        segs (list): a list of segmented masks for every frame with statistics for the initalization frames.
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
        recent_frames (3D numpy.ndarray of float32, shape=(Lt,Lx,Ly)): the images from the last "Lt" frames.
            Theese images are after spatial fitering but before temporal filtering.
    '''

    [Lx, Ly]=dims
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)
    (rowsnb, colsnb) = dimsnb

    # PreProcessing
    network_input, med_frame3 = preprocess_complete(bb, dimspad, dimsnb, network_input, med_frame2, \
        Poisson_filt, mask2, bf, fft_object_b, fft_object_c, median_decimate=1, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=True)
    # network_input = network_input[:, :rowsnb, :colsnb]
    # med_frame3 = med_frame3[:, :rowsnb, :colsnb]
    if useTF==True: # store the reacent frames, used in futher temporal filtering.
        recent_frames = bb[-(Poisson_filt.size):, :rowsnb, :colsnb]
    else:
        recent_frames = None

    # Network inference
    network_input = np.expand_dims(network_input[:, :rowspad, :colspad], axis=-1)
    prob_map = fff.predict(network_input, batch_size=batch_size_init)
    prob_map = prob_map.squeeze()[:, :Lx, :Ly]
    
    # Threshold the probablity map to binary
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)

    # PostProcessing for every frame
    minArea = Params_post['minArea']
    avgArea = Params_post['avgArea']
    if useMP:
        if p is None:
            p = mp.Pool()
        segs_init = p.starmap(separate_neuron, [(frame, None, minArea, avgArea, useWT) for frame in pmaps_b], chunksize=1)
    else:
        segs_init =[separate_neuron(frame, None, minArea, avgArea, useWT) for frame in pmaps_b]

    return med_frame3, segs_init, recent_frames

