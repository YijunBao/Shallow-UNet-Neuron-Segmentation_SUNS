# %%
import os
import math
import numpy as np
import time
import h5py
import sys
import pyfftw
from scipy import signal
from scipy import special

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import par1
import preprocessing_functions
from par3 import fastthreshold
from seperate_multi import separateNeuron


def load_wisdom_txt(dir_wisdom):
    '''Load the learned wisdom files.

    Inputs: 
        dir_wisdom (dir): the folder of the saved wisdom files.

    Outputs:
        cc: learned wisdom. Return None if the files do not exist.
    '''
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


def plan_fft3(frames_init, dims1, prealloc=True):
    '''Plan FFT for pyfftw for a 3D video.

    Inputs: 
        frames_init (int): Number of images.
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the image.
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        bb: array of the real video.
        bf: array of the complex spectrum.
        fft_object_b: Object for forward FFT.
        fft_object_c: Object for inverse FFT.
    '''
    (rows1, cols1) = dims1
    bb = pyfftw.zeros_aligned((frames_init, rows1, cols1), dtype='float32', n=8)
    if prealloc:
        bf = pyfftw.zeros_aligned((frames_init, rows1, cols1//2+1), dtype='complex64', n=8)
    else:
        bf = pyfftw.empty_aligned((frames_init, rows1, cols1//2+1), dtype='complex64', n=8)
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
    return bb, bf, fft_object_b, fft_object_c


def plan_fft2(dims1):
    '''Plan FFT for pyfftw for a 2D image.

    Inputs: 
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the image.

    Outputs:
        bb: array to store the real image.
        bf: array to store the complex spectrum.
        fft_object_b: Object for forward FFT.
        fft_object_c: Object for inverse FFT.
    '''
    (rows1, cols1) = dims1
    bb = pyfftw.zeros_aligned((rows1, cols1), dtype='float32', n=8)
    # No pre-allocation, because this step is not executed in the initialization stage,
    # So the running time counts to the total time.
    bf = pyfftw.empty_aligned((rows1, cols1//2+1), dtype='complex64', n=8)
    fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
    fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
    return bb, bf, fft_object_b, fft_object_c


def plan_mask2(dims, dims1, gauss_filt_size):
    '''Calculate the 2D mask for spatial filtering.

    Inputs: 
        dims (tuplel of int, shape = (2,)): lateral dimension of the image.
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the padded image.
        gauss_filt_size (float): The standard deviation of the spatial Gaussian filter in pixels

    Outputs:
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
    '''
    (rows, cols) = dims
    (rows1, cols1) = dims1
    # gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
    # mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
    # mask1_col = signal.gaussian(cols1+1, gauss_filt_kernel)[:-1]
    gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
    mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
    gauss_filt_kernel = cols/2/math.pi/gauss_filt_size
    mask1_col = signal.gaussian(cols1+1, gauss_filt_kernel)[:-1]
    mask2 = np.outer(mask1_row, mask1_col)
    mask = np.fft.fftshift(mask2)
    mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
    return mask2


def init_online(bb, dims, network_input, pmaps_b, fff, thresh_pmap_float, Params_post, med_frame2=None, mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, Poisson_filt=np.array([1]), \
        useSF=True, useTF=True, useSNR=True, useWT=False, batch_size_init=1, p=None):

    [Lx, Ly]=dims
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)
    network_input, med_frame3 = preprocess_init(bb, dimspad, network_input, med_frame2, Poisson_filt, mask2, \
        bf, fft_object_b, fft_object_c, useSF=useSF, useTF=useTF, useSNR=useSNR)
    if useTF==True:
        recent_frames = network_input[-(Poisson_filt.size):]
    else:
        recent_frames = None

    # %% Network inference
    network_input = np.expand_dims(network_input, axis=-1)
    prob_map = fff.predict(network_input, batch_size=batch_size_init)

    prob_map = prob_map.squeeze()[:, :Lx, :Ly]
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)

    # %% PostProcessing
    # Masks_2 = complete_segment(pmaps_b, Params_post, p=p, useWT=useWT)
    segs_init = segment_init(pmaps_b, Params_post, p=p, useWT=useWT)

    return med_frame3, segs_init, recent_frames


def preprocess_init(bb, dimspad, network_input=None, med_frame2=None, Poisson_filt=np.array([1]), mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, useSF=False, useTF=True, useSNR=True, prealloc=True):
    '''Pre-process the initial part of a video into an SNR video.

    Inputs: 
        bb: array storing the raw video.
        dimspad (tuplel of int, shape = (2,)): lateral dimension of the padded image.
        network_input (3D empty numpy.ndarray of float32): empty array to store the SNR video.
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        bf(empty, default to None): array to store the complex spectrum for FFT.
        fft_object_b(default to None): Object for forward FFT.
        fft_object_c(default to None): Object for inverse FFT.
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (3D numpy.ndarray of float32): the SNR video.
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
    '''
    (rowspad, colspad) = dimspad
    # %% Homomorphic spatial filtering
    if useSF:
        preprocessing_functions.spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2, display=False)
        # par1.fastlog(bb)
        # fft_object_b()
        # par1.fastmask(bf, mask2)
        # fft_object_c()
        # par1.fastexp(bb)

    # %% Temporal filtering
    if useTF:
        preprocessing_functions.temporal_filtering(bb[:, :rowspad, :colspad], network_input, Poisson_filt, display=False)
        # par1.fastconv(bb[:, :rowspad, :colspad], network_input, Poisson_filt)
    else:
        network_input = bb[:, :rowspad, :colspad]

    # %% Median computation and normalization
    # result = np.copy(network_input.transpose([1, 2, 0]))
    if useSNR:
        med_frame3 = preprocessing_functions.SNR_normalization(
            network_input, med_frame2, (rowspad, colspad), 1, display=True)
        # par1.fastquant(result, np.array([0.5, 0.25], dtype='float32'), med_frame2)
        # temp_noise = (med_frame2[:, :, 0]-med_frame2[:, :, 1])/(math.sqrt(2)*special.erfinv(0.5))
        # zeronoise = (temp_noise==0)
        # if np.any(zeronoise):
        #     [x0, y0] = zeronoise.nonzero()
        #     for (x,y) in zip(x0, y0):
        #         new_noise = np.std(network_input[:, x, y])
        #         if new_noise>0:
        #             temp_noise[x, y] = new_noise
        #         else:
        #             temp_noise[x, y] = np.inf

        # med_frame2[:, :, 1] = np.reciprocal(temp_noise).astype('float32')
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing
        # par1.fastnormf(network_input, med_frame3)
    else:
        med_frame3 = preprocessing_functions.median_normalization(
            network_input, med_frame2, (rowspad, colspad), 1, display=True)
        # par1.fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
        # par1.fastnormback(network_input, 0, med_frame2[:,:,0].mean())
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing

    return network_input, med_frame3


def segment_init(pmaps: np.ndarray, Params: dict, useMP=True, useWT=False, p=None):
    '''The input probablity map must be pre-thresholded'''
    # nframes = len(pmaps)
    # (Lx, Ly) = pmaps[0].shape
    minArea = Params['minArea']
    avgArea = Params['avgArea']
    # thresh_pmap = Params['thresh_pmap']
    # thresh_mask = Params['thresh_mask']
    # thresh_COM0 = Params['thresh_COM0']
    # thresh_COM = Params['thresh_COM']
    # thresh_IOU = Params['thresh_IOU']
    # thresh_consume = Params['thresh_consume']
    # cons = Params['cons']
    # if useMP: # %% Run segmentation with multiprocessing
    #     p = mp.Pool(mp.cpu_count())

    if useMP: # %% Run segmentation with multiprocessing
        segs = p.starmap(separateNeuron, [(frame, None, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
    else: # %% Run segmentation without multiprocessing
        segs = []
        nframes = pmaps.shape[0]
        for ind in range(nframes):
            segs.append(separateNeuron(pmaps[ind], None, minArea, avgArea, useWT)) #, eng
    # num_neurons = sum([x[1].size for x in segs])

    return segs#, Masks_2 # masks_final_2, times_final #, matname_output

