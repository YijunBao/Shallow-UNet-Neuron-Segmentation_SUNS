import cv2
import math
import numpy as np
import time
from scipy import signal
from scipy import special
import multiprocessing as mp
import pyfftw
import h5py
import fissa
import os
from scipy.io import loadmat

import par1


def process_video(dir_video:str, Exp_ID:str, Params:dict, 
        dir_network_input:str = None, useSF=False, useTF=True, useSNR=True, prealloc=True):
    '''Pre-process the registered a video into an SNR video.

    Inputs: 
        dir_video (str): The folder containing the input video.
        Exp_ID (str): The filer name of the input video. 
            The file must be a ".h5" file, with dataset "mov" being the input video (shape = (T0,Lx0,Ly0)).
        Params (dict): Parameters for pre-processing.
            Params['gauss_filt_size'] (float): The standard deviation of the spatial Gaussian filter in pixels
            Params['Poisson_filt'] (numpy.ndarray of float32, 1D): The temporal filter kernel
            Params['num_median_approx'] (int): Number of frames used to compute 
                the median and median-based standard deviation
            Params['nn'] (int): Number of frames at the beginning of the video to be processed.
                The remaining video is not considered a part of the input video.
        dir_network_input (str): The folder to save the SNR video (network_input) in hard drive.
            If dir_network_input == None, then the SNR video is not stored in hard drive
        useSF (bool): True if spatial filtering is used.
        useSF (bool): True if temporal filtering is used.
        useSF (bool): True if pixel-by-pixel SNR normalization filtering is used.
        useSF (bool): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.

    Outputs:
        network_input (numpy.ndarray of float32): the SNR video obtained after pre-processing. 
            The shape is (T,Lx,Ly), where T is shorter than T0 due to temporal filtering, 
            and Lx and Ly are Lx0 and Ly0 padded to multiples of 8, so that the images can be process by the shallow U-Net.
        start (float): the starting time of the pipline (after the data is loaded into memory)
        In addition, the SNR video is saved in "dir_network_input"
    '''
    nn = Params['nn']
    if useSF:
        gauss_filt_size = Params['gauss_filt_size']
    if useTF:
        Poisson_filt = Params['Poisson_filt']
    if useSNR:
        num_median_approx = Params['num_median_approx']

    h5_video = dir_video + Exp_ID + '.h5'
    h5_file = h5py.File(h5_video,'r')
    (nframes, rows, cols) = h5_file['mov'].shape
    # Make the lateral number of pixels a multiple of 8, so that the CNN can process them 
    rowspad = math.ceil(rows/8)*8 
    colspad = math.ceil(cols/8)*8
    # Only keep the first "nn" frames to process
    nframes = min(nframes,nn)
    
    if useSF:
        # %% FFT planning
        rows1 = cv2.getOptimalDFTSize(rows)
        cols1 = cv2.getOptimalDFTSize(cols)
        print(rows, cols, nn, '->', rows1, cols1, nn)

        start_plan = time.time()
        try: # if the learned wisdom files have been saved, load them. Otherwise, learn wisdom later
            Length_data=str((nn, rows1, cols1))
            file = open('wisdom\\'+Length_data+'x1.txt', 'rb')
            cc = file.read()
            file.close()
            file = open('wisdom\\'+Length_data+'x2.txt', 'rb')
            dd = file.read()
            file.close()
            file = open('wisdom\\'+Length_data+'x3.txt', 'rb')
            ee = file.readline()
            cc = (cc, dd, ee)
            pyfftw.import_wisdom(cc)
        except:
            pass

        bb = pyfftw.zeros_aligned((nn, rows1, cols1), dtype='float32', n=8)
        if prealloc:
            bf = pyfftw.zeros_aligned((nn, rows1, cols1//2+1), dtype='complex64', n=8)
        else:
            bf = pyfftw.empty_aligned((nn, rows1, cols1//2+1), dtype='complex64', n=8)
        fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
        fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
        end_plan = time.time()
        print('FFT planning: {} s'.format(end_plan - start_plan))

        # %% Initialization: Calculate the spatial filter and set variables.
        gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
        mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
        gauss_filt_kernel = cols/2/math.pi/gauss_filt_size
        mask1_col = signal.gaussian(cols1+1, gauss_filt_kernel)[:-1]
        mask = np.outer(mask1_row, mask1_col)
        mask = np.fft.fftshift(mask)
        mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
    else:
        end_plan = time.time()
        bb=np.zeros((nframes, rowspad, colspad), dtype='float32')

    # %% Initialization: Set variables.
    if useTF:
        leng_tf = Poisson_filt.size
        nframesf = nframes - leng_tf + 1 # Number of frames after temporal filtering
        if prealloc:
            network_input = np.ones((nframesf, rowspad, colspad), dtype='float32')
        else:
            network_input = np.zeros((nframesf, rowspad, colspad), dtype='float32')
    else:
        nframesf = nframes
    if prealloc:
        med_frame2 = np.ones((rows, cols, 2), dtype='float32')
    else:
        med_frame2 = np.zeros((rows, cols, 2), dtype='float32')
    end_init = time.time()
    print('Initialization: {} s'.format(end_init - end_plan))
    
    # %% Load the raw video into "bb"
    for t in range(nframes): # use this one to save memory
        bb[t, :rows, :cols] = np.array(h5_file['mov'][t])
    h5_file.close()
    end_load = time.time()
    print('data loading: {} s'.format(end_load - end_init))

    start = time.time() # The pipline starts after the video is loaded into memory
    # %% Homomorphic spatial filtering based on FFT
    if useSF:
        par1.fastlog(bb)
        f1start = time.time()
        fft_object_b()
        maskstart = time.time()
        par1.fastmask(bf, mask2)
        f2start = time.time()
        fft_object_c()
        expstart = time.time()
        par1.fastexp(bb)
        endhomofilt = time.time()
        if not prealloc:
            del bf, fft_object_c, fft_object_b
        print('spatial filtering: {} s'.format(endhomofilt-start))
        print('    Log time:', f1start-start, 's')
        print('    FFT1 time:', maskstart-f1start, 's')
        print('    Mask time:', f2start-maskstart, 's')
        print('    FFT2 time:', expstart-f2start, 's')
        print('    Exp time:', endhomofilt-expstart, 's')
    else:
        endhomofilt = time.time()

    # %% Temporal filtering
    if useTF:
        par1.fastconv(bb[:nframes, :rows, :cols], network_input[:, :rows, :cols], Poisson_filt)
    else:
        network_input = bb[:nframes, :rowspad, :colspad]
    if not prealloc:
        del bb
    endtempfilt = time.time()
    if useTF:
        print('temporal filtering: {} s'.format(endtempfilt-endhomofilt))

    # %% Median computation and SNR normalization
    median_decimate = max(1,nframes//num_median_approx)
    result = np.copy(network_input[::median_decimate, :rows, :cols].transpose([1, 2, 0]))
    if useSNR:
        par1.fastquant(result, np.array([0.5, 0.25], dtype='float32'), med_frame2)
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
                new_noise = np.std(network_input[:, x, y])
                if new_noise>0:
                    temp_noise[x, y] = new_noise
                else:
                    temp_noise[x, y] = np.inf

        # med_frame2[:, :, 1] stores the median-based standard deviation
        med_frame2[:, :, 1] = np.reciprocal(temp_noise).astype('float32')
        med_frame3 = np.copy(med_frame2.transpose([2,0,1])) # Using copy to avoid computer crashing
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastnormf(network_input[:, :rows, :cols], med_frame3)
    else:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
        par1.fastnormback(network_input[:, :rows, :cols], med_frame2[:,:,0].mean())
    if not prealloc:
        del result
    endnormtime = time.time()
    print('normalization: {} s'.format(endnormtime - endmedtime))

    end = time.time()
    print('total per frame: {} ms'.format((end - start) / nframes *1000))

    # if "dir_network_input" is not None, save network_input to an ".h5" file
    if dir_network_input:
        f = h5py.File(dir_network_input+Exp_ID+".h5", "w")
        f.create_dataset("network_input", data = network_input)
        f.close()
        end_saving = time.time()
        print('Network_input saving: {} s'.format(end_saving - end))
        
    return network_input, start


def generate_masks(network_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame.

    Inputs: 
        network_input (numpy.ndarray of float32): the SNR video obtained after pre-processing
            (shape = (T,Lx,Ly)).
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a ".h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
        In addition, the raw and unmixed traces before and after FISSA are saved in the same folder
            but a different sub-folder in another ".h5" file. The ".h5" file has two datasets, 
            "raw_traces" and "unmixed_traces" saving the traces before and after FISSA, respectively. 
    '''
    try: # If the ".mat" file is saved in '-v7.3' format
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')
        mat.close()
    except OSError: # If the ".mat" file is not saved in '-v7.3' format
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0]).astype('bool')
    (nframesf, rowspad, colspad) = network_input.shape
    (ncells, rows, cols) = rois.shape
    # The lateral shape of "network_input" can be larger than that of "rois" due to padding in pre-processing
    # This step crop "network_input" to match the shape of "rois"
    network_input = network_input[:, :rows, :cols]

    # Use FISSA to calculate the decontaminated traces of neural activities. 
    folder = 'FISSA'    
    start = time.time()
    experiment = fissa.Experiment([network_input], [rois.tolist()], folder)
    experiment.separation_prep(redo=True)
    prep = time.time()
    
    experiment.separate(redo_prep=False, redo_sep=True)
    finish = time.time()
    experiment.save_to_matlab()
    del network_input
    print('FISSA time: {} s'.format(finish-start))
    print('    Preparation time: {} s'.format(prep-start))
    print('    Separation time: {} s'.format(finish-prep))

    # %% Extract raw and unmixed traces from the output of FISSA
    raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
    del experiment

    # Save the raw and unmixed traces into a ".h5" file under folder "dir_trace".
    dir_trace = dir_save+"traces\\"
    if not os.path.exists(dir_trace):
        os.makedirs(dir_trace)        
    f = h5py.File(dir_trace+Exp_ID+".h5", "w")
    f.create_dataset("raw_traces", data = raw_traces)
    f.create_dataset("unmixed_traces", data = unmixed_traces)
    f.close()

    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = dir_save+"temporal_masks({})\\".format(thred_ratio)
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(dir_temporal_masks+Exp_ID+".h5", "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))


def generate_masks_from_traces(file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame.
        This version generate temporal masks from the traces saved in "generate_masks", 
        so it does not redo FISSA and does not need input video.

    Inputs: 
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a ".h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
    '''
    try: # If the ".mat" file is saved in '-v7.3' format
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')
        mat.close()
    except OSError: # If the ".mat" file is not saved in '-v7.3' format
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0])
    (_, rows, cols) = rois.shape

    # %% Extract raw and unmixed traces from the saved ".h5" file
    dir_trace = dir_save+"traces\\"
    f = h5py.File(dir_trace+Exp_ID+".h5", "r")
    raw_traces = np.array(f["raw_traces"])
    unmixed_traces = np.array(f["unmixed_traces"])
    f.close()
    nframesf = unmixed_traces.shape[1]
    
    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = dir_save+"temporal_masks({})\\".format(thred_ratio)
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(dir_temporal_masks+Exp_ID+".h5", "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))

