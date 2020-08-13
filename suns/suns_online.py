# %%
import os
import cv2
import math
import numpy as np
import time
import h5py
import sys
import pyfftw
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
sys.path.insert(1, '..\\Online')
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from shallow_unet import get_shallow_unet
import functions_online
import functions_init
import par_online
from combine import segs_results, uniqueNeurons2_simp, group_neurons, piece_neurons_IOU, piece_neurons_consume


def suns_online(filename_video, filename_CNN, Params_pre, Params_post, dims, \
        frames_init, merge_every, batch_size_init=1, useSF=True, useTF=True, useSNR=True, \
        useWT=False, show_intermediate=True, prealloc=True, display=True, useMP=True, p=None):
    '''The complete SUNS online procedure.

    Inputs: 
        filename_video (str): The path of the file of the input video.
            The file must be a ".h5" file, with dataset "mov" being the input video (shape = (T0,Lx0,Ly0)).
        filename_CNN (str): The path of the CNN model. 
        Params_pre (dict): Parameters for pre-processing.
            Params_pre['gauss_filt_size'] (float): The standard deviation of the spatial Gaussian filter in pixels
            Params_pre['Poisson_filt'] (1D numpy.ndarray of float32): The temporal filter kernel
            Params_pre['num_median_approx'] (int): Number of frames used to compute 
                the median and median-based standard deviation
            Params_pre['nn'] (int): Number of frames at the beginning of the video to be processed.
                The remaining video is not considered a part of the input video.
        Params_post (dict): Parameters for post-processing.
            Params_post['minArea']: Minimum area of a valid neuron mask (unit: pixels).
            Params_post['avgArea']: The typical neuron area (unit: pixels).
            Params_post['thresh_pmap']: The probablity threshold. Values higher than thresh_pmap are active pixels. 
                It is stored in uint8, so it should be converted to float32 before using.
            Params_post['thresh_mask']: Threashold to binarize the real-number mask.
            Params_post['thresh_COM0']: Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_post['thresh_COM']: Threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_post['thresh_IOU']: Threshold of IOU used for merging neurons.
            Params_post['thresh_consume']: Threshold of consume ratio used for merging neurons.
            Params_post['cons']: Minimum number of consecutive frames that a neuron should be active for.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw video.
        frames_init (int): Number of frames used for initialization.
        merge_every (int): SUNS online merge the newly segmented frames every "merge_every" frames.
        batch_size_init (int, default to 1): batch size of CNN inference for initialization frames.
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        show_intermediate (bool, default to True): Indicator of whether 
            consecutive frame requirement is applied to screen neurons after every update. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.
        display (bool, default to True): Indicator of whether to show intermediate information
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        Masks (3D numpy.ndarray of bool, shape = (n,Lx,Ly)): the final segmented masks. 
        Masks_2 (scipy.csr_matrix of bool, shape = (n,Lx*Ly)): the final segmented masks in the form of sparse matrix. 
        time_total (list of float, shape = (3,)): the total time spent 
            for initalization, online processing, and total processing
        time_frame (list of float, shape = (3,)): the average time spent on every frame
            for initalization, online processing, and total processing
    '''
    if display:
        start = time.time()
    (Lx, Ly) = dims
    # zero-pad the lateral dimensions to multiples of 8, suitable for CNN
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    dimspad = (rowspad, colspad)

    Poisson_filt = Params_pre['Poisson_filt']
    gauss_filt_size = Params_pre['gauss_filt_size']
    nn = Params_pre['nn']
    leng_tf = Poisson_filt.size
    leng_past = 2*leng_tf # number of past frames stored for temporal filtering
    list_time_per = np.zeros(nn)

    # Load CNN model
    fff = get_shallow_unet()
    fff.load_weights(filename_CNN)
    # run CNN inference once to warm up
    init_imgs = np.zeros((batch_size_init, Lx, Ly, 1), dtype='float32')
    init_masks = np.zeros((batch_size_init, Lx, Ly, 1), dtype='uint8')
    fff.evaluate(init_imgs, init_masks, batch_size=batch_size_init)
    del init_imgs, init_masks

    # load optimal post-processing parameters
    minArea = Params_post['minArea']
    avgArea = Params_post['avgArea']
    # thresh_pmap = Params_post['thresh_pmap']
    thresh_mask = Params_post['thresh_mask']
    thresh_COM0 = Params_post['thresh_COM0']
    thresh_COM = Params_post['thresh_COM']
    thresh_IOU = Params_post['thresh_IOU']
    thresh_consume = Params_post['thresh_consume']
    cons = Params_post['cons']
    thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    # thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version


    # Spatial filtering preparation
    if useSF==True:
        # lateral dimensions slightly larger than the raw video but faster for FFT
        rows1 = cv2.getOptimalDFTSize(rowspad)
        cols1 = cv2.getOptimalDFTSize(colspad)
        
        # if the learned 2D and 3D wisdom files have been saved, load them. 
        # Otherwise, learn wisdom later
        Length_data2=str((rows1, cols1))
        cc2 = functions_init.load_wisdom_txt('wisdom\\'+Length_data2)
        
        Length_data3=str((frames_init, rows1, cols1))
        cc3 = functions_init.load_wisdom_txt('wisdom\\'+Length_data3)
        if cc3:
            pyfftw.import_wisdom(cc3)

        # mask for spatial filter
        mask2 = functions_init.plan_mask2(dims, (rows1, cols1), gauss_filt_size)
        # FFT planning
        (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft3(frames_init, (rows1, cols1))
    else:
        (mask2, bf, fft_object_b, fft_object_c) = (None, None, None, None)
        bb=np.zeros((frames_init, rowspad, colspad), dtype='float32')

    # Temporal filtering preparation
    frames_initf = frames_init - leng_tf + 1
    if useTF==True:
        if prealloc:
            # past frames stored for temporal filtering
            past_frames = np.ones((leng_past, rowspad, colspad), dtype='float32')
        else:
            past_frames = np.zeros((leng_past, rowspad, colspad), dtype='float32')
    else:
        past_frames = None
    
    if prealloc: # Pre-allocate memory for some future variables
        med_frame2 = np.ones((rowspad, colspad, 2), dtype='float32')
        video_input = np.ones((frames_initf, rowspad, colspad), dtype='float32')        
        pmaps_b_init = np.ones((frames_initf, Lx, Ly), dtype='uint8')        
        frame_SNR = np.ones(dimspad, dtype='float32')
        pmaps_b = np.ones(dims, dtype='uint8')
    else:
        med_frame2 = np.zeros((rowspad, colspad, 2), dtype='float32')
        video_input = np.zeros((frames_initf, rowspad, colspad), dtype='float32')        
        pmaps_b_init = np.zeros((frames_initf, Lx, Ly), dtype='uint8')        
        frame_SNR = np.zeros(dimspad, dtype='float32')
        pmaps_b = np.zeros(dims, dtype='uint8')

    if display:
        time_init = time.time()
        print('Parameter initialization time: {} s'.format(time_init-start))


    # %% Load raw video
    h5_img = h5py.File(filename_video, 'r')
    video_raw = np.array(h5_img['mov'])
    h5_img.close()
    nframes = video_raw.shape[0]
    nframesf = nframes - leng_tf + 1
    bb[:, :Lx, :Ly] = video_raw[:frames_init]
    if display:
        time_load = time.time()
        print('Load data: {} s'.format(time_load-time_init))


    # %% Actual processing starts after the video is loaded into memory
    # Initialization using the first "frames_init" frames
    print('Initialization of algorithms using the first {} frames'.format(frames_init))
    if display:
        start_init = time.time()
    med_frame3, segs_all, recent_frames = functions_init.init_online(
        bb, dims, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
        med_frame2, mask2, bf, fft_object_b, fft_object_c, Poisson_filt, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, useWT=useWT, batch_size_init=batch_size_init, p=p)
    if useTF==True:
        past_frames[:leng_tf] = recent_frames
    tuple_temp = functions_online.merge_complete(segs_all[:frames_initf], dims, Params_post)
    if show_intermediate:
        Masks_2 = functions_online.select_cons(tuple_temp)

    if display:
        end_init = time.time()
        time_init = end_init-start_init
        time_frame_init = time_init/(frames_initf)*1000
        print('Initialization time: {:6f} s, {:6f} ms/frame'.format(time_init, time_frame_init))


    if display:
        start_online = time.time()
    # Spatial filtering preparation for online processing. 
    # Attention: this part counts to the total time
    if useSF:
        if cc2:
            pyfftw.import_wisdom(cc2)
        (bb, bf, fft_object_b, fft_object_c) = functions_init.plan_fft2((rows1, cols1))
    else:
        (bf, fft_object_b, fft_object_c) = (None, None, None)
        bb=np.zeros(dimspad, dtype='float32')
    
    print('Start frame by frame processing')
    # %% Online processing for the following frames
    current_frame = leng_tf
    t_merge = frames_initf
    for t in range(frames_initf,nframesf):
        if display:
            start_frame = time.time()
        # load the current frame
        bb[:Lx, :Ly] = video_raw[t+leng_tf-1]
        bb[Lx:, :] = 0
        bb[:, Ly:] = 0
        
        # PreProcessing
        frame_SNR = functions_online.preprocess_online(bb, dimspad, med_frame3, frame_SNR, \
            past_frames[current_frame-leng_tf:current_frame], mask2, bf, fft_object_b, fft_object_c, \
            Poisson_filt, useSF=useSF, useTF=useTF, useSNR=useSNR)

        # CNN inference
        frame_prob = functions_online.CNN_online(frame_SNR, fff, dims)

        # first step of post-processing
        segs = functions_online.postprocess_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)
        segs_all.append(segs)

        # temporal merging 1: combine neurons with COM distance smaller than thresh_COM0
        if ((t + 1 - t_merge) == merge_every) or (t==nframesf-1):
            # uniques, times_uniques = uniqueNeurons1_simp(segs_all[t_merge:], thresh_COM0) # minArea,
            totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs_all[t_merge:])
            uniques, times_uniques = uniqueNeurons2_simp(totalmasks, neuronstate, COMs, \
                areas, probmapID, minArea=0, thresh_COM0=thresh_COM0)

        # temporal merging 2: combine neurons with COM distance smaller than thresh_COM
        if ((t - 0 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                groupedneurons, times_groupedneurons = \
                    group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques)

        # temporal merging 3: combine neurons with IoU larger than thresh_IOU
        if ((t - 1 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                piecedneurons_1, times_piecedneurons_1 = \
                    piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)

        # temporal merging 4: combine neurons with conumse ratio larger than thresh_consume
        if ((t - 2 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                piecedneurons, times_piecedneurons = \
                    piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
                # masks of new neurons
                masks_add = piecedneurons
                # indices of frames when the neurons are active
                times_add = [np.unique(x) + t_merge for x in times_piecedneurons]
                    
                # Refine neurons using consecutive occurence
                if masks_add.size:
                    # new real-number masks
                    masks_add = [x for x in masks_add]
                    # new binary masks
                    Masksb_add = [(x >= x.max() * thresh_mask).astype('float') for x in masks_add]
                    # areas of new masks
                    area_add = np.array([x.nnz for x in Masksb_add])
                    # indicators of whether the new masks satisfy consecutive frame requirement
                    have_cons_add = functions_online.refine_seperate_cons(times_add, cons)
                else:
                    Masksb_add = []
                    area_add = np.array([])
                    have_cons_add = np.array([])

            else: # does not find any active neuron
                Masksb_add = []
                masks_add = []
                times_add = times_uniques
                area_add = np.array([])
                have_cons_add = np.array([])
            tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)

        # temporal merging 5: merge newly found neurons within the recent "merge_every" frames with existing neurons
        if ((t - 3 - t_merge) == merge_every) or (t==nframesf-1):
            tuple_temp = functions_online.merge_2(tuple_temp, tuple_add, dims, Params_post)
            t_merge += merge_every
            if show_intermediate:
                Masks_2 = functions_online.select_cons(tuple_temp)

        current_frame +=1
        # Update the stored latest frames when it runs out: move them "leng_tf" ahead
        if current_frame >= leng_past:
            current_frame = leng_tf
            past_frames[:leng_tf] = past_frames[-leng_tf:]
        if display:
            end_frame = time.time()
            list_time_per[t] = end_frame - start_frame
        if t % 1000 == 0:
            print('{} frames has been processed'.format(t))

    if not show_intermediate:
        Masks_2 = functions_online.select_cons(tuple_temp)
    # final result. Masks_2 is a 2D sparse matrix of the segmented neurons
    if len(Masks_2):
        Masks_2 = sparse.vstack(Masks_2)
    else:
        Masks_2 = sparse.csc_matrix((0,dims[0]*dims[1]))

    if display:
        end_online = time.time()
        time_online = end_online-start_online
        time_frame_online = time_online/(nframesf-frames_initf)*1000
        print('Online time: {:6f} s, {:6f} ms/frame'.format(time_online, time_frame_online))

    # Save total processing time, and average processing time per frame
    if display:
        end_final = time.time()
        time_all = end_final-start_init
        time_frame_all = time_all/nframes*1000
        print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
        time_total = np.array([time_init, time_online, time_all])
        time_frame = np.array([time_frame_init, time_frame_online, time_frame_all])
    else:
        time_total = np.zeros((3,))
        time_frame = np.zeros((3,))

    # convert to a 3D array of the segmented neurons
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly))
    return Masks, Masks_2, time_total, time_frame
