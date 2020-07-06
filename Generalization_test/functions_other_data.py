# %%
import os
import numpy as np
import json
# import tifffile
from skimage.external import tifffile
import glob
import sys

import cv2
import math
# import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy import special
from scipy import sparse
from scipy.io import loadmat
import multiprocessing as mp
import pyfftw
import h5py
import nibabel as nib

sys.path.insert(0, '..\\PreProcessing')
sys.path.insert(0, '..\\Network')
import par1


def tomask(coords, dims):
    mask = np.zeros(dims, dtype='bool')
    for coor in coords:
        mask[coor[0], coor[1]] = True
    return mask


def data_info_neurofinder(fname_info):
    # fname_info = 'E:\\NeuroFinder\\train videos\\neurofinder.' + Exp_ID + '\\info.json'
    f = open(fname_info)
    info = json.load(f)
    if '01.0' in fname_info:
        magnificatioin = 0.78 / info['pixels-per-micron']
    else:
        magnificatioin = 0.78 * info['pixels-per-micron']
    # frame_rate = info['rate-hz']
    # dims = info['dimensions']
    # indicator = info['indicator']
    return info, magnificatioin  # frame_rate, mag, dims


def data_info_caiman(fname_info):
    # fname_info = 'E:\\CaImAn data\\WEBSITE\\' + Exp_ID + '\\info.json'
    f = open(fname_info)
    info = json.load(f)
    # frame_rate = info['rate-hz']
    # decay = info['decay_time']
    # dims = info['dimensions']
    Exp_ID = fname_info.split('\\')[-2]
    if Exp_ID[0:2] == 'N.':
        if Exp_ID[-2:] == '.t':
            fname_info_neurofinder = 'E:\\NeuroFinder\\test videos\\neurofinder.' + Exp_ID[2:7] + '.test\\info.json'
            _, magnificatioin = data_info_neurofinder(fname_info_neurofinder)
        else:
            fname_info_neurofinder = 'E:\\NeuroFinder\\train videos\\neurofinder.' + Exp_ID[2:7] + '\\info.json'
            _, magnificatioin = data_info_neurofinder(fname_info_neurofinder)
    else:
        magnificatioin = info['radius']/8*1.15*0.78
    return info, magnificatioin  # frame_rate, decay, dims


def temporal_filter_exp(info):
    indicator = info['indicator']
    frame_rate = info['rate-hz']
    # decay = info['decay_time']
    if indicator == 'GCaMP6f':
        # rise = 0.018
        decay = 0.2
    elif indicator == 'GCaMP6s':
        # rise = 0.072
        # decay = 0.8
        decay = 1
    leng_tf = np.ceil(frame_rate*decay)+1
    Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay)
    Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
    return Poisson_filt


def temporal_filter(info):
    indicator = info['indicator']
    frame_rate = info['rate-hz']
    # decay = info['decay_time']
    if indicator == 'GCaMP6f':
        fs_template = 30
        h5f = h5py.File('GCaMP6f_spike_tempolate_mean.h5','r')
    elif indicator == 'GCaMP6s':
        fs_template = 3
        h5f = h5py.File('GCaMP6s_spike_tempolate_mean.h5','r')
    Poisson_template = np.array(h5f['filter_tempolate']).squeeze()
    h5f.close()
    peak = Poisson_template.argmax()
    length = Poisson_template.shape
    xp = np.arange(-peak,length-peak,1)/fs_template
    x = np.arange(np.round(-peak*frame_rate/fs_template), np.round(length-peak*frame_rate/fs_template), 1)/frame_rate
    Poisson_filt = np.interp(x,xp,Poisson_template)
    Poisson_filt = Poisson_filt[Poisson_filt>=np.exp(-1)].astype('float32')
    # Poisson_filt = Poisson_filt[Poisson_filt>=0.47].astype('float32')
    return Poisson_filt


# load images into a video
def load_caiman_video(dir_images, Exp_ID):
    files = glob.glob(dir_images + Exp_ID + '\\*.tif*')
    images = [tifffile.imread(image) for image in files]
    # images = []
    # dims = tifffile.imread(files[0]).shape
    # for image in files:
    #     temp=np.zeros((513,513), dtype='float32')
    #     temp[:dims[0], :dims[1]] = tifffile.imread(image)
    #     images.append(temp)

    if len(images[0].shape)==2: 
        video = np.array(images)
    elif len(images[0].shape)==3: 
        video = np.concatenate(images, axis=0)
    # dims = images[0].shape[-2:]
    # del images
    return video


# load the regions 
def load_caiman_roi(dir_roi, Exp_ID, dims):
    regions_filename = dir_roi + Exp_ID + '\\regions\\consensus_regions.json'
    with open(regions_filename) as f:
        regions = json.load(f)
    masks = np.array([tomask(s['coordinates'], dims) for s in regions])
    return masks


# load the regions 
def load_caiman_roi2(dir_roi, Exp_ID, dims):
    regions_filename = dir_roi + Exp_ID + '\\regions\\consensus_regions.json'
    with open(regions_filename) as f:
        regions = json.load(f)
    masks2 = sparse.vstack([sparse.csr_matrix(tomask(s['coordinates'], dims).ravel()) for s in regions])
    return masks2


# load the regions 
def load_neurofinder_roi2(dir_roi, Exp_ID):
    filename_GTMask = dir_roi + Exp_ID[1] + Exp_ID[3:5] + '.mat'
    try:
        f = h5py.File(filename_GTMask, "r")
        masks = np.array(f["FinalMasks"]).astype('bool')
        f.close()
    except OSError: 
        f = loadmat(filename_GTMask)
        masks = np.array(f["FinalMasks"]).transpose([2,1,0])
    dims = masks.shape
    masks2 = sparse.csr_matrix(masks.reshape((dims[0], dims[1]*dims[2])))
    return masks2


def process_video_prealloc_others(dir_images:str, Exp_ID:str, Params:dict
        , dir_network_input:str = None, useSF=False, useTF=True, useSNR=True, isonline=False):
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

    # h5_video = dir_video + Exp_ID + '.h5'
    # h5_file = h5py.File(h5_video,'r')
    # (nframes, rows, cols) = h5_file['mov'].shape
    if 'CaImAn' in dir_images:
        dataset_type = 'caiman'
        files = glob.glob(dir_images + Exp_ID + '\\*.tif*')
        image0 = tifffile.imread(files[0])
        dims = image0.shape
        (rows, cols) = dims[-2:]
        ndims = len(dims)
        nframes = len(files)
        if ndims==3:
            nframes_per = dims[0]
            nframes = nframes * nframes_per
        raw_dims = (nframes, rows, cols)
    elif 'NeuroFinder' in dir_images:
        dataset_type = 'neurofinder'
        # fname = dir_images + Exp_ID + '_processed.nii'
        # fnii = nib.load(fname)
        # (cols, rows, nframes) = fnii.shape
        # raw_dims = (nframes, rows, cols)
        h5_video = dir_images + Exp_ID + '.h5'
        h5_file = h5py.File(h5_video,'r')
        raw_dims = (nframes, rows, cols) = h5_file['mov'].shape

    # files = glob.glob(dir_images + 'neurofinder.' + Exp_ID + '\\images\\*.tif*')
    rowspad = math.ceil(rows/8)*8
    colspad = math.ceil(cols/8)*8

    # %% FFT planning
    if useSF:
        nframes = min(nframes,nn)
        nn = nframes
        rows1 = cv2.getOptimalDFTSize(rows)
        cols1 = cv2.getOptimalDFTSize(cols)
        print(rows, cols, nn, '->', rows1, cols1, nn)

        # %% FFT planning
        start_plan = time.time()
        # Length_data=str((nn, rows1, cols1))
        # file = open('wisdom\\'+Length_data+'x1.txt', 'rb')
        # cc = file.read()
        # file.close()
        # file = open('wisdom\\'+Length_data+'x2.txt', 'rb')
        # dd = file.read()
        # file.close()
        # file = open('wisdom\\'+Length_data+'x3.txt', 'rb')
        # ee = file.readline()
        # cc = (cc, dd, ee)
        # pyfftw.import_wisdom(cc)

        bb = pyfftw.zeros_aligned((nn, rows1, cols1), dtype='float32', n=8)
        bf = pyfftw.zeros_aligned((nn, rows1, cols1//2+1), dtype='complex64', n=8)
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
        mask2 = np.outer(mask1_row, mask1_col)
        # mask2 = np.outer(mask1_row, mask1_row)
        mask2 = np.fft.ifftshift(mask2) #/ mask2.sum() * mask2.size
        # mask2_s = np.real(np.fft.ifft(mask2))
        # mask2 = mask2 / mask2_s.sum()
        # mask_delta = np.ones((rows1, cols1//2+1))
        # mask_delta[0,0] = 1
        # mask = (mask_delta-mask2[:, :cols1//2+1]).astype('float32')
        # mask = np.zeros((rows1, cols1//2+1), dtype='float32')
        mask = (1-mask2[:, :cols1//2+1]).astype('float32')
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
    # bb[:nframes, :rows, :cols] = video # np.array(h5_file['mov'][:nframes])
    if dataset_type == 'caiman':
        if ndims==2:
            for (k, image) in enumerate(files):
                bb[k, :rows, :cols] = tifffile.imread(image)
        elif ndims==3:
            for (k, image) in enumerate(files):
                bb[k*nframes_per:(k+1)*nframes_per, :rows, :cols] = tifffile.imread(image)
    elif dataset_type == 'neurofinder':
        # fname = dir_images + Exp_ID + '_processed.nii'
        # fnii = nib.load(fname)
        # bb[:nframes, :rows, :cols] = fnii.get_fdata().transpose([2,1,0])
        bb[:nframes, :rows, :cols] = np.array(h5_file['mov']) #[:nframes]
    # h5_file.close()
    end_load = time.time()
    print('data loading: {} s'.format(end_load - end_init))

    start = time.time()
    # %% Homomorphic spatial filtering
    if useSF:
        par1.fastlog(bb)
        f1start = time.time()
        fft_object_b()
        maskstart = time.time()
        par1.fastmask(bf, mask)
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
        # par1.fastconv(bb[:nframes, :rows, :cols], video_input, Poisson_filt)
        bbs=bb[:nframes, :rows, :cols].copy()
        par1.fastconv(bbs, video_input, Poisson_filt)
        # par1.slowconv(bb[:nframes, :rows, :cols], video_input, Poisson_filt)
    else:
        video_input = bb[:nframes, :rowspad, :colspad]
    # del bb
    endtempfilt = time.time()
    if useTF:
        print('temporal filtering: {} s'.format(endtempfilt-endhomofilt))

    # %% Median computation and normalization
    if isonline:
        result = np.copy(video_input[:num_median_approx, :rows, :cols].transpose([1, 2, 0]))
    else:
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
        
    return network_input, start, raw_dims


# # %%
# if __name__ == '__main__':
#     Exp_ID = 'N.03.00.t'
#     dir_images = 'E:\\CaImAn data\\images_'
#     video =load_caiman_video(dir_images, Exp_ID)
#     dims = video.shape[-2:]
#     dir_roi = 'E:\\CaImAn data\\WEBSITE\\'
#     masks = load_caiman_roi(dir_roi, dims, Exp_ID)

