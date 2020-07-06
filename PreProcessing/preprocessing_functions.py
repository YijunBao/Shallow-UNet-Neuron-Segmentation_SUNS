import cv2
import math
import numpy as np
# import matplotlib.pyplot as plt
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
# import par1_online as par1

def process_video(dir_video:str, Exp_ID:str, Params:dict, 
        dir_network_input:str = None, useSF=False, useTF=True, useSNR=True):
    num_median_approx = Params['num_median_approx']
    nn = Params['nn']
    if useSF:
        gauss_filt_size = Params['gauss_filt_size']  # signa in pixels
    if useTF:
        # frame_rate = Params['frame_rate']
        # decay = Params['decay']
        # leng_tf = Params['leng_tf']
        Poisson_filt = Params['Poisson_filt']
        # Poisson_filt = np.array([1], dtype='float32')
    if useSNR:
        network_baseline = Params['network_baseline']
        network_SNRscale = Params['network_SNRscale']

    h5_video = dir_video + Exp_ID + '.h5'
    h5_file = h5py.File(h5_video,'r')
    (nframes, rows, cols) = h5_file['mov'].shape
    rowspad = math.ceil(rows/8)*8
    colspad = math.ceil(cols/8)*8
    
    if useSF:
        # %% FFT planning
        rows1 = cv2.getOptimalDFTSize(rows)
        cols1 = cv2.getOptimalDFTSize(cols)
        print(rows, cols, nn, '->', rows1, cols1, nn)

        start_plan = time.time()
        try:
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
        bf = pyfftw.empty_aligned((nn, rows1, cols1//2+1), dtype='complex64', n=8)
        fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
        fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
        end_plan = time.time()
        print('FFT planning: {} s'.format(end_plan - start_plan))

        # %% Initialization
        gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
        mask1_row = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
        gauss_filt_kernel = cols/2/math.pi/gauss_filt_size
        mask1_col = signal.gaussian(cols1+1, gauss_filt_kernel)[:-1]
        # mask1_col = signal.gaussian(rows1+1, gauss_filt_kernel)[:-1]
        mask = np.outer(mask1_row, mask1_col)
        # gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
        # mask = np.outer(signal.gaussian(rows1, gauss_filt_kernel), signal.gaussian(cols1, gauss_filt_kernel))
        mask = np.fft.fftshift(mask)
        # mask2 = np.zeros((rows1, cols1//2+1), dtype='float32')
        mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
    else:
        end_plan = time.time()
        bb=np.zeros((nframes, rowspad, colspad), dtype='float32')

    if useTF:
        # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
        # Poisson_filt = Poisson_filt / Poisson_filt.sum()
        leng_tf = Poisson_filt.size
        nframesf = nframes - leng_tf + 1
        video_input = np.zeros((nframesf, rowspad, colspad), dtype='float32')
    else:
        nframesf = nframes
    # med_frame = np.zeros((rows, cols), dtype='float32')
    med_frame2 = np.zeros((rows, cols, 2), dtype='float32')
    end_init = time.time()
    print('Initialization: {} s'.format(end_init - end_plan))
    
    # %% Load data
    # bb[:nframes, :rows, :cols] = np.array(h5_file['mov'])
    for t in range(nframes): # use this one to save memory
        bb[t, :rows, :cols] = np.array(h5_file['mov'][t])
    h5_file.close()
    end_load = time.time()
    print('data loading: {} s'.format(end_load - end_init))

    start = time.time()
    # %% Homomorphic spatial filtering
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
        par1.fastconv(bb[:nframes, :rows, :cols], video_input[:, :rows, :cols], Poisson_filt)
    else:
        video_input = bb[:nframes, :rowspad, :colspad]
    del bb
    endtempfilt = time.time()
    if useTF:
        print('temporal filtering: {} s'.format(endtempfilt-endhomofilt))

    # %% Median computation and normalization
    median_decimate = max(1,nframes//num_median_approx) #20
    result = np.copy(video_input[::median_decimate, :rows, :cols].transpose([1, 2, 0]))
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

        med_frame2[:, :, 1] = (network_SNRscale) * np.reciprocal(temp_noise).astype('float32')
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1]))
        med_frame3 = med_frame2.transpose([2, 0, 1])
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastnormf(video_input[:, :rows, :cols], med_frame3, network_baseline)
    else:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
        par1.fastnormback(video_input[:, :rows, :cols], 0, med_frame2[:,:,0].mean())
    del result
    endnormtime = time.time()
    print('normalization: {} s'.format(endnormtime - endmedtime))

    # network_input = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
    # par1.fastclip(video_input, network_input)
    # network_input = np.zeros((nframesf, rowspad, colspad), dtype='float32')
    # network_input[:nframesf, :rows, :cols] = video_input
    # rows_topad = rowspad - rows
    # cols_topad = colspad - cols
    # network_input = np.pad(video_input, ((0,0),(0,rows_topad),(0,cols_topad)),'constant', constant_values=(0, 0))
    # par1.fastclipf(video_input, -2, 6)
    network_input = video_input#.astype('float16')

    end = time.time()
    # print('clipping: {} s'.format(end - endnormtime))
    # print('total per frame: {} ms'.format((end - start) / nframes *1000))

    if dir_network_input:
        f = h5py.File(dir_network_input+Exp_ID+".h5", "w")
        f.create_dataset("network_input", data = network_input)
        f.close()
        end_saving = time.time()
        print('Network_input saving: {} s'.format(end_saving - end))
        
    return video_input, start


def process_video_prealloc(dir_video:str, Exp_ID:str, Params:dict, 
        dir_network_input:str = None, useSF=False, useTF=True, useSNR=True):
    nn = Params['nn']
    if useSF:
        gauss_filt_size = Params['gauss_filt_size']  # signa in pixels
    if useTF:
        # frame_rate = Params['frame_rate']
        # decay = Params['decay']
        # leng_tf = Params['leng_tf']
        Poisson_filt = Params['Poisson_filt']
        # Poisson_filt = np.array([1], dtype='float32')
    if useSNR:
        num_median_approx = Params['num_median_approx']
        network_baseline = Params['network_baseline']
        network_SNRscale = Params['network_SNRscale']

    h5_video = dir_video + Exp_ID + '.h5'
    h5_file = h5py.File(h5_video,'r')
    (nframes, rows, cols) = h5_file['mov'].shape
    # rowspad = math.ceil(rows/8)*8
    # colspad = math.ceil(cols/8)*8
    rowspad = math.ceil(rows/16)*16
    colspad = math.ceil(cols/16)*16
    nframes = min(nframes,nn)

    if useSF:
        rows1 = cv2.getOptimalDFTSize(rows)
        cols1 = cv2.getOptimalDFTSize(cols)
        print(rows, cols, nn, '->', rows1, cols1, nn)

        # %% FFT planning
        start_plan = time.time()
        try:
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
        bf = pyfftw.zeros_aligned((nn, rows1, cols1//2+1), dtype='complex64', n=8)
        fft_object_b = pyfftw.FFTW(bb, bf, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads=mp.cpu_count())
        fft_object_c = pyfftw.FFTW(bf, bb, axes=(-2, -1), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads=mp.cpu_count())
        end_plan = time.time()
        print('FFT planning: {} s'.format(end_plan - start_plan))

        # %% Initialization
        gauss_filt_kernel = rows/2/math.pi/gauss_filt_size
        mask = np.outer(signal.gaussian(rows1, gauss_filt_kernel), signal.gaussian(cols1, gauss_filt_kernel))
        mask = np.fft.fftshift(mask)
        # mask2 = np.zeros((rows1, cols1//2+1), dtype='float32')
        mask2 = (1-mask[:, :cols1//2+1]).astype('float32')
        # med_frame = np.zeros((rows, cols), dtype='float32')
    else:
        end_plan = time.time()
        bb=np.zeros((nframes, rowspad, colspad), dtype='float32')

    # %% Initialization
    if useTF:
        # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
        # Poisson_filt = Poisson_filt / Poisson_filt.sum()  
        leng_tf = Poisson_filt.size
        nframesf = nframes - leng_tf + 1
        video_input = np.ones((nframesf, rowspad, colspad), dtype='float32')
    else:
        nframesf = nframes
    med_frame2 = np.ones((rows, cols, 2), dtype='float32')
    end_init = time.time()
    print('Initialization: {} s'.format(end_init - end_plan))
    
    # %% Load data
    # bb[:nframes, :rows, :cols] = np.array(h5_file['mov'][:nframes])
    for t in range(nframes): # use this one to save memory
        bb[t, :rows, :cols] = np.array(h5_file['mov'][t])
    h5_file.close()
    end_load = time.time()
    print('data loading: {} s'.format(end_load - end_init))

    start = time.time()
    # %% Homomorphic spatial filtering
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
        # del bf, fft_object_c, fft_object_b
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
        par1.fastconv(bb[:nframes, :rows, :cols], video_input, Poisson_filt)
    else:
        video_input = bb[:nframes, :rowspad, :colspad]
    # del bb
    endtempfilt = time.time()
    if useTF:
        print('temporal filtering: {} s'.format(endtempfilt-endhomofilt))

    # %% Median computation and normalization
    median_decimate = max(1,nframes//num_median_approx) #20
    result = np.copy(video_input[::median_decimate, :rows, :cols].transpose([1, 2, 0]))
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

        med_frame2[:, :, 1] = (network_SNRscale) * np.reciprocal(temp_noise).astype('float32')
        # med_frame3 = np.copy(med_frame2.transpose([2,0,1]))
        med_frame3 = med_frame2.transpose([2, 0, 1])
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastnormf(video_input[:, :rows, :cols], med_frame3, network_baseline)
    else:
        endmedtime = time.time()
        print('median computation: {} s'.format(endmedtime - endtempfilt))
        par1.fastquant(result, np.array([0.5], dtype='float32'), med_frame2[:,:,0:1])
        par1.fastnormback(video_input[:, :rows, :cols], 0, med_frame2[:,:,0].mean())
    # del result
    endnormtime = time.time()
    print('normalization: {} s'.format(endnormtime - endmedtime))

    # network_input = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
    # par1.fastclip(video_input, network_input)
    # par1.fastclipf(video_input, -2, 6)
    network_input = video_input#.astype('float16')
    end = time.time()
    # print('clipping: {} s'.format(end - endnormtime))
    print('total per frame: {} ms'.format((end - start) / nframes *1000))

    if dir_network_input:
        f = h5py.File(dir_network_input+Exp_ID+".h5", "w")
        f.create_dataset("network_input", data = network_input)
        f.close()
        end_saving = time.time()
        print('Network_input saving: {} s'.format(end_saving - end))
        
    return network_input, start


def generate_masks(video_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    try:
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')#[:13]
        mat.close()
    except OSError: 
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0]).astype('bool')
    # ncells = rois
    (nframesf, rowspad, colspad) = video_input.shape
    (ncells, rows, cols) = rois.shape
    video_input = video_input[:, :rows, :cols]
    # rowspad = math.ceil(rows/8)*8
    # colspad = math.ceil(cols/8)*8

    folder = 'FISSA'    
    start = time.time()
    experiment = fissa.Experiment([video_input], [rois.tolist()], folder)
    experiment.separation_prep(redo=True)
    prep = time.time()
    
    experiment.separate(redo_prep=False, redo_sep=True)
    finish = time.time()
    experiment.save_to_matlab()
    del video_input
    print('FISSA time: {} s'.format(finish-start))
    print('    Preparation time: {} s'.format(prep-start))
    print('    Separation time: {} s'.format(finish-prep))

    # %% Extract raw and unmixed traces    
    raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
    del experiment
    dir_trace = dir_save+"traces\\"
    if not os.path.exists(dir_trace):
        os.makedirs(dir_trace)        
    f = h5py.File(dir_trace+Exp_ID+".h5", "w")
    f.create_dataset("raw_traces", data = raw_traces)
    f.create_dataset("unmixed_traces", data = unmixed_traces)
    f.close()
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %%
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        dir_temporal_masks = dir_save+"temporal_masks({})\\".format(thred_ratio)
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(dir_temporal_masks+Exp_ID+".h5", "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))


def generate_masks_from_traces(file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    try:
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')#[:13]
        mat.close()
    except OSError: 
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0])
    # ncells = rois
    # (nframesf, rowspad, colspad) = video_input.shape
    (ncells, rows, cols) = rois.shape
    # video_input = video_input[:, :rows, :cols]
    rowspad = math.ceil(rows/8)*8
    colspad = math.ceil(cols/8)*8

    # %% Extract raw and unmixed traces
    dir_trace = dir_save+"traces\\"
    f = h5py.File(dir_trace+Exp_ID+".h5", "r")
    raw_traces = np.array(f["raw_traces"])
    unmixed_traces = np.array(f["unmixed_traces"])
    f.close()
    nframesf = unmixed_traces.shape[1]
    
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %%
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        dir_temporal_masks = dir_save+"temporal_masks({})\\".format(thred_ratio)
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(dir_temporal_masks+Exp_ID+".h5", "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))

