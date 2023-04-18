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
from cpuinfo import get_cpu_info

from suns.PreProcessing.par1 import fastexp, fastmask, fastlog, \
    fastconv, fastquant, fastnormf, fastnormback, fastmediansubtract


def find_dataset(h5_file): 
    '''Automatically find the dataset name in the h5 file containing the input movie.

    Inputs: 
        h5_file (h5py.File): the input h5 file containing the movie.

    Outputs:
        dset (str): the dataset name in the h5 file containing the input movie.
    '''
    list_dset = list(h5_file.keys())
    num_dset = len(list_dset)
    if num_dset == 1:
        dset = list_dset[0]
    else:
        list_size = np.zeros(num_dset, dtype='uint64')
        for (ind,dset) in enumerate(list_dset):
            if isinstance(h5_file[dset], h5py.Dataset):
                list_size[ind] = h5_file[dset].size
        ind_dset = list_size.argmax()
        if list_size[ind_dset] > 0:
            dset = list_dset[ind_dset]
        else:
            raise(KeyError('Cannot find any dataset. Please do not put the movie under any group'))
    return dset


def load_wisdom_txt(dir_wisdom):
    '''Load the learned wisdom files. This speeds up FFT planing

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


def export_wisdom_txt(cc, dir_wisdom):
    '''Export the learned wisdom files. This speeds up FFT planing

    Inputs: 
        cc: learned wisdom. Return None if the files do not exist.
        dir_wisdom (dir): the folder of the saved wisdom files.

    Outputs: None
    '''
    if not os.path.exists(dir_wisdom):
        os.makedirs(dir_wisdom) 

    file = open(dir_wisdom+'x1.txt', "wb")
    file.write(cc[0])
    file.close
    file = open(dir_wisdom+'x2.txt', "wb")
    file.write(cc[1])
    file.close
    file = open(dir_wisdom+'x3.txt', "wb")
    file.write(cc[2])
    file.close()


def plan_fft(frames_init, dims1, prealloc=True):
    '''Plan FFT for pyfftw for a 3D video.

    Inputs: 
        frames_init (int): Number of images.
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the image.
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        bb(3D numpy.ndarray of float32): array of the real video.
        bf(3D numpy.ndarray of complex64): array of the complex spectrum.
        fft_object_b(pyfftw.FFTW): Object for forward FFT.
        fft_object_c(pyfftw.FFTW): Object for inverse FFT.
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


def plan_mask2(dims, dims1, gauss_filt_size):
    '''Calculate the 2D mask for spatial filtering. 
        It is the Fourier trainsform of a 2D Gaussian funcition with standard deviation of "gauss_filt_size". 

    Inputs: 
        dims (tuple of int, shape = (2,)): lateral dimension of the image.
        dims1 (tuple of int, shape = (2,)): lateral dimension of the padded image.
        gauss_filt_size (float): The standard deviation of the spatial Gaussian filter in pixels

    Outputs:
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
    '''
    (rows, cols) = dims
    (rows1, cols1) = dims1
    gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
    mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
    gauss_filt_kernel = cols/2/math.pi/gauss_filt_size
    mask1_col = signal.gaussian(cols1+1, gauss_filt_kernel)[:-1]
    mask2 = np.outer(mask1_row, mask1_col)
    mask = np.fft.fftshift(mask2)
    mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
    return mask2


def spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2, display=False):
    '''Apply spatial homomorphic filtering to the input video.
        bb = exp(IFFT(mask2*FFT(log(bb+1)))).

    Inputs: 
        bb(3D numpy.ndarray of float32): array storing the raw video.
        bf(3D numpy.ndarray of complex64): array to store the complex spectrum for FFT.
        fft_object_b(pyfftw.FFTW): Object for forward FFT.
        fft_object_c(pyfftw.FFTW): Object for inverse FFT.
        mask2 (2D numpy.ndarray of float32): 2D mask for spatial filtering.
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        No output, but "bb" is changed to the spatially filtered video during the function
    '''
    if display:
        start = time.time()

    fastlog(bb)
    if display:
        f1start = time.time()

    fft_object_b()
    if display:
        maskstart = time.time()

    fastmask(bf, mask2)
    if display:
        f2start = time.time()

    fft_object_c()
    if display:
        expstart = time.time()

    fastexp(bb)
    if display:
        endhomofilt = time.time()
        print('spatial filtering: {} s'.format(endhomofilt-start))
        print('    Log time:', f1start-start, 's')
        print('    FFT1 time:', maskstart-f1start, 's')
        print('    Mask time:', f2start-maskstart, 's')
        print('    FFT2 time:', expstart-f2start, 's')
        print('    Exp time:', endhomofilt-expstart, 's')


def temporal_filtering(bb, network_input, Poisson_filt=np.array([1]), display=False):
    '''Apply temporal filtering to the input video.
        It convolves the input video "bb" with the flipped version of "Poisson_filt".

    Inputs: 
        bb(3D numpy.ndarray of float32): array storing the raw video.
        network_input (3D empty numpy.ndarray of float32): empty array to store the temporally filtered video.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        No output, but "network_input" is changed to the temporally filtered video during the function
    '''
    if display:
        start = time.time()
    fastconv(bb, network_input, Poisson_filt)
    if display:
        endtempfilt = time.time()
        print('temporal filtering: {} s'.format(endtempfilt-start))


def median_calculation(network_input, med_frame2=None, median_decimate=1, display=True):
    '''Calculate the median and median-based standard deviation of a video.

    Inputs: 
        network_input (numpy.ndarray of float32): the input video. 
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        median_decimate (int, default to 1): Median and median-based standard deviation are 
            calculate from every "median_decimate" frames of the video
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
    '''
    if display:
        start = time.time()
    (_, rows, cols) = network_input.shape
    if med_frame2 is None:
        med_frame2 = np.zeros(rows, cols, 2, dtype='float32')

    result = np.copy(network_input[::median_decimate].transpose([1, 2, 0]))
    fastquant(result[:rows, :cols, :], np.array([0.5, 0.25], dtype='float32'), med_frame2[:rows, :cols, :])
    # med_frame2[:, :, 0] stores the median
    
    # Noise is estimated using median-based standard deviation calculated from 
    # the difference bwtween 0.5 quantile and 0.25 quantile
    temp_noise = (med_frame2[:rows, :cols, 0]-med_frame2[:rows, :cols, 1])/(math.sqrt(2)*special.erfinv(0.5))
    zeronoise = (temp_noise==0)
    # if the calculated temp_noise is 0 at some pixels, replace the median-based standard deviation 
    # with conventional stantard deviation
    if np.any(zeronoise):
        [x0, y0] = zeronoise.nonzero()
        for (x,y) in zip(x0, y0):
            new_noise = np.std(network_input[:, x, y])
            if new_noise>0:
                temp_noise[x, y] = new_noise
            else:
                temp_noise[x, y] = np.inf

    # med_frame2[:, :, 1] stores the median-based standard deviation
    med_frame2[:rows, :cols, 1] = np.reciprocal(temp_noise).astype('float32')
    # med_frame3 = med_frame2.transpose([2,0,1])
    med_frame3 = np.copy(med_frame2[:rows, :cols, :].transpose([2,0,1])) # Using copy to avoid computer crashing
    if display:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - start))

    return med_frame3


def SNR_normalization(network_input, med_frame2=None, median_decimate=1, display=True):
    '''Normalize the video to be an SNR video.
        network_input(t,x,y) = (network_input(t,x,y) - median(x,y)) / (median_based_standard_deviation(x,y)).

    Inputs: 
        network_input (numpy.ndarray of float32): the input video. 
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        median_decimate (int, default to 1): Median and median-based standard deviation are 
            calculate from every "median_decimate" frames of the video
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
        In addition, "network_input" is changed to the SNR video during the function
    '''
    if display:
        start = time.time()

    med_frame3 = median_calculation(network_input, med_frame2, median_decimate, display=False)
    if display:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - start))

    fastnormf(network_input, med_frame3)
    if display:
        endnormtime = time.time()
        print('normalization: {} s'.format(endnormtime - endmedtime))
    return med_frame3


def median_normalization(network_input, med_frame2=None, median_decimate=1, display=True):
    '''Normalize the video by dividing to its temporal median.
        network_input(t,x,y) = network_input(t,x,y) / mean(median(x,y)).

    Inputs: 
        network_input (numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video. 
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median.
        median_decimate (int, default to 1): Median is 
            calculate from every "median_decimate" frames of the video
        display (bool, default to True): Indicator of whether display the timing information

    Outputs:
        med_frame3 (3D numpy.ndarray of float32): the median.
        In addition, "network_input" is changed to become the normalized video during the function
    '''
    if display:
        start = time.time()
    (_, rows, cols) = network_input.shape
    if med_frame2 is None:
        med_frame2 = np.zeros(rows, cols, 2, dtype='float32')

    result = np.copy(network_input[::median_decimate].transpose([1, 2, 0]))
    fastquant(result[:rows, :cols, :], np.array([0.5], dtype='float32'), med_frame2[:rows, :cols,0:1])
    # med_frame2[:, :, 0] stores the median
    if display:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - start))

    fastnormback(network_input[:, :rows, :cols], max(1, med_frame2[:rows, :cols, 0].mean()))
    # med_frame3 = med_frame2.transpose([2,0,1])
    med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing
    if display:
        endnormtime = time.time()
        print('normalization: {} s'.format(endnormtime - endmedtime))
    return med_frame3
    

def preprocess_complete(bb, dimspad, dimsnb, network_input=None, med_frame2=None, Poisson_filt=np.array([1]), mask2=None, \
        bf=None, fft_object_b=None, fft_object_c=None, median_decimate=1, \
        useSF=False, useTF=True, useSNR=True, med_subtract=False, prealloc=True, display=False):
    '''Pre-process the input video "bb" into an SNR video "network_input".
        It applies spatial filter, temporal filter, and SNR normalization in sequance. 
        Each step is optional.

    Inputs: 
        bb (3D numpy.ndarray of float32, shape = (T0,Lx0f,Ly0f)): array storing the raw video.
        dimspad (tuplel of int, shape = (2,)): lateral dimension of the padded images to be multiples of 8.
        dimsnb (tuplel of int, shape = (2,)): lateral dimension of the padded images for numba calculation in pre-processing.
        network_input (3D empty numpy.ndarray of float32, shape = (T,Lx,Ly)): empty array to store the SNR video.
        med_frame2 (3D empty numpy.ndarray of float32, default to None): 
            empty array to store the median and median-based standard deviation.
        Poisson_filt (1D numpy.ndarray of float32, default to np.array([1])): The temporal filter kernel
        bf (empty, default to None): array to store the complex spectrum for FFT.
        fft_object_b (default to None): Object for forward FFT.
        fft_object_c (default to None): Object for inverse FFT.
        median_decimate (int, default to 1): Median and median-based standard deviation are 
            calculate from every "median_decimate" frames of the video
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (3D numpy.ndarray of float32): the SNR video.
        med_frame3 (3D numpy.ndarray of float32): the median and median-based standard deviation.
    '''

    (rowspad, colspad) = dimspad
    (rowsnb, colsnb) = dimsnb
    
    if useSF: # Homomorphic spatial filtering based on FFT
        spatial_filtering(bb, bf, fft_object_b, fft_object_c, mask2, display=display)
    # if not prealloc:
    #     del bf

    if useTF: # Temporal filtering
        temporal_filtering(bb[:, :rowspad, :colspad], network_input[:, :rowspad, :colspad], Poisson_filt, display=display)
        # use copy() to enhance the stability when the lateral size of bb is 512.
        # temporal_filtering(bb[:, :rowsnb, :colsnb].copy(), network_input, Poisson_filt, display=display)
    else:
        network_input = bb[:, :rowsnb, :colsnb]

    if med_subtract: # and not useSF: # Subtract every frame with its median.
        if display:
            start = time.time()
        temp = np.zeros(network_input.shape[:2], dtype = 'float32')
        fastmediansubtract(network_input[:, :rowspad, :colspad], temp[:, :rowspad], 2)
        if display:
            endmedsubtr = time.time()
            print('median subtraction: {} s'.format(endmedsubtr-start))

    # Median computation and normalization
    if useSNR:
        med_frame3 = SNR_normalization(
            network_input[:, :rowspad, :colspad], med_frame2, median_decimate, display=display)
    else:
        med_frame3 = median_normalization(
            network_input[:, :rowspad, :colspad], med_frame2, median_decimate, display=display)

    return network_input, med_frame3


def preprocess_video(dir_video:str, Exp_ID:str, Params:dict, 
        dir_network_input:str = None, useSF=False, useTF=True, useSNR=True, \
        med_subtract=False, prealloc=True, display=True):
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
        dir_network_input (str, default to None): The folder to save the SNR video (network_input) in hard drive.
            If dir_network_input == None, then the SNR video is not stored in hard drive
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing. 
            The shape is (T,Lx,Ly), where T is shorter than T0 due to temporal filtering, 
            and Lx and Ly are Lx0 and Ly0 padded to multiples of 8, so that the images can be process by the shallow U-Net.
        start (float): the starting time of the pipline (after the data is loaded into memory)
        In addition, the SNR video is saved in "dir_network_input" as a "(Exp_ID).h5" file containing a dataset "network_input".
    '''
    if useSF:
        gauss_filt_size = Params['gauss_filt_size']
    Poisson_filt = Params['Poisson_filt']
    num_median_approx = Params['num_median_approx']

    h5_video = os.path.join(dir_video, Exp_ID + '.h5')
    h5_file = h5py.File(h5_video,'r')
    dset = find_dataset(h5_file)
    (nframes, rows, cols) = h5_file[dset].shape
    # Make the lateral number of pixels a multiple of 8, so that the CNN can process them 
    rowspad = math.ceil(rows/8)*8 
    colspad = math.ceil(cols/8)*8
    dimspad = (rowspad, colspad)
    # Only keep the first "nn" frames to process
    if 'nn' in Params.keys():
        nn = Params['nn']
        nframes = min(nframes,nn)
    else:
        nn = nframes
    
    rowsnb = rowspad
    colsnb = colspad
    if 'win' in sys.platform.lower() and 'AMD' in get_cpu_info()['brand_raw'].upper():
        # Avoid crashing
        if rowspad % 256 == 0:
            rowsnb = rowspad + 1
        if colspad % 256 == 0:
            colsnb = colspad + 1
    dimsnb = (rowsnb, colsnb)

    if useSF:
        # lateral dimensions slightly larger than the raw video but faster for FFT
        rowsfft = cv2.getOptimalDFTSize(rowsnb)
        colsfft = cv2.getOptimalDFTSize(colsnb)

        if display:
            print(rows, cols, nn, '->', rowsfft, colsfft, nn)
            start_plan = time.time()
        # if the learned wisdom files have been saved, load them. Otherwise, learn wisdom later
        Length_data=str((nn, rowsfft, colsfft))
        cc = load_wisdom_txt(os.path.join('wisdom', Length_data))
        if cc:
            export = False
            pyfftw.import_wisdom(cc)
        else:
            export = True

        # FFT planning
        bb, bf, fft_object_b, fft_object_c = plan_fft(nn, (rowsfft, colsfft), prealloc)
        if export:
            cc = pyfftw.export_wisdom()
            export_wisdom_txt(cc, os.path.join('wisdom', Length_data))

        if display:
            end_plan = time.time()
            print('FFT planning: {} s'.format(end_plan - start_plan))

        # %% Initialization: Calculate the spatial filter and set variables.
        mask2 = plan_mask2((rows, cols), (rowsfft, colsfft), gauss_filt_size)
    else:
        bb=np.zeros((nframes, rowsnb, colsnb), dtype='float32')
        (mask2, bf, fft_object_b, fft_object_c) = (None, None, None, None)
        end_plan = time.time()

    # %% Initialization: Set variables.
    if useTF:
        leng_tf = Poisson_filt.size
        nframesf = nframes - leng_tf + 1 # Number of frames after temporal filtering
    else:
        nframesf = nframes
    if prealloc:
        network_input = np.ones((nframesf, rowsnb, colsnb), dtype='float32')
        med_frame2 = np.ones((rowsnb, colsnb, 2), dtype='float32')
    else:
        network_input = np.zeros((nframesf, rowsnb, colsnb), dtype='float32')
        med_frame2 = np.zeros((rowsnb, colsnb, 2), dtype='float32')
    median_decimate = max(1,nframes//num_median_approx)
    if display:
        end_init = time.time()
        print('Initialization: {} s'.format(end_init - end_plan))
    
    # %% Load the raw video into "bb"
    for t in range(nframes): # use this one to save memory
        bb[t, :rows, :cols] = np.clip(np.array(h5_file[dset][t]), 0, None)
    h5_file.close()
    if display:
        end_load = time.time()
        print('data loading: {} s'.format(end_load - end_init))

    start = time.time() # The pipline starts after the video is loaded into memory
    network_input, _ = preprocess_complete(bb, dimspad, dimsnb, network_input, med_frame2, Poisson_filt, mask2, \
        bf, fft_object_b, fft_object_c, median_decimate, useSF, useTF, useSNR, med_subtract, prealloc, display)
    network_input = network_input[:, :rowspad, :colspad]

    if display:
        end = time.time()
        print('total per frame: {} ms'.format((end - start) / nframes *1000))

    # if "dir_network_input" is not None, save network_input to an ".h5" file
    if dir_network_input:
        if not os.path.exists(dir_network_input):
            os.makedirs(dir_network_input) 
        f = h5py.File(os.path.join(dir_network_input, Exp_ID+".h5"), "w")
        f.create_dataset("network_input", data = network_input)
        f.close()
        if display:
            end_saving = time.time()
            print('Network_input saving: {} s'.format(end_saving - end))
        
    return network_input, start

