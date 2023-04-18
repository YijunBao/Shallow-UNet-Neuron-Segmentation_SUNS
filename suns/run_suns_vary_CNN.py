# %%
import sys
import os
import cv2
import math
import numpy as np
import time
import h5py
import pyfftw
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

# sys.path.insert(1, '../..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.Online.functions_online import merge_2, merge_2_nocons, merge_complete, select_cons, \
    preprocess_online, CNN_online, separate_neuron_online, refine_seperate_cons_online
from suns.Online.functions_init import init_online, plan_fft2
from suns.PreProcessing.preprocessing_functions import preprocess_video, \
    plan_fft, plan_mask2, load_wisdom_txt, SNR_normalization, median_normalization
from suns.Network.shallow_unet import get_shallow_unet_more, get_shallow_unet_more_equal
from suns.PostProcessing.par3 import fastthreshold
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume
from suns.PostProcessing.complete_post import complete_segment


def suns_batch(dir_video, Exp_ID, filename_CNN, Params_pre, Params_post, dims, \
        batch_size_eval=1, useSF=True, useTF=True, useSNR=True, med_subtract=False, \
        useWT=False, prealloc=True, display=True, useMP=True, p=None,\
        n_depth=3, n_channel=4, skip=[1], activation='elu', double=True):
    '''The complete SUNS batch procedure.
        It uses the trained CNN model from "filename_CNN" and the optimized hyper-parameters in "Params_post"
        to process the video "Exp_ID" in "dir_video"

    Inputs: 
        dir_video (str): The folder containing the input video.
            Each file must be a ".h5" file, with dataset "mov" being the input video (shape = (T0,Lx0,Ly0)).
        Exp_ID (str): The filer name of the input raw video. 
        filename_CNN (str): The path of the trained CNN model. 
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
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        useWT (bool, default to False): Indicator of whether watershed is used. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.
        display (bool, default to True): Indicator of whether to show intermediate information
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 
        n_depth (int, default to 3): Number of resolution depth. Can be 2, 3, or 4
        n_channel (int, default to 4): Number of channels per feature map in the first resolution depth.
            For deeper depths, the numbers of feature map will double for every depth. 
        skip (list of int, default to [1]]): Indeces of resolution depths that use skip connections.
            The shallowest depth is 1, and 1 should usually be in "skip".
        activation (str, default to 'elu): activation function. Can be 'elu' or 'relu'.
        double (bool, default to False): True if the number of channels per featrue map doubled in every resolution depth. 

    Outputs:
        Masks (3D numpy.ndarray of bool, shape = (n,Lx0,Ly0)): the final segmented masks. 
        Masks_2 (scipy.csr_matrix of bool, shape = (n,Lx0*Ly0)): the final segmented masks in the form of sparse matrix. 
        time_total (list of float, shape = (4,)): the total time spent 
            for pre-processing, CNN inference, post-processing, and total processing
        time_frame (list of float, shape = (4,)): the average time spent on every frame
            for pre-processing, CNN inference, post-processing, and total processing
    '''
    if display:
        start = time.time()
    (Lx, Ly) = dims
    rowspad = math.ceil(Lx/8)*8
    colspad = math.ceil(Ly/8)*8
    # load CNN model
    if double:
        get_shallow_unet = get_shallow_unet_more
    else:
        get_shallow_unet = get_shallow_unet_more_equal

    fff = get_shallow_unet(size=None, n_depth=n_depth, n_channel=n_channel, skip=skip, activation=activation)
    fff.load_weights(filename_CNN)
    # run CNN inference once to warm up
    init_imgs = np.zeros((batch_size_eval, rowspad, colspad, 1), dtype='float32')
    init_masks = np.zeros((batch_size_eval, rowspad, colspad, 1), dtype='uint8')
    fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)
    del init_imgs, init_masks

    # thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version
    if display:
        time_init = time.time()
        print('Initialization time: {} s'.format(time_init-start))

    # %% Actual processing starts after the video is loaded into memory
        # which is in the middle of "preprocess_video", represented by the output "start"
    # pre-processing including loading data
    video_input, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc, display=display)
    nframes = video_input.shape[0]
    if display:
        end_pre = time.time()
        time_pre = end_pre-start
        time_frame_pre = time_pre/nframes*1000
        print('Pre-Processing time: {:6f} s, {:6f} ms/frame'.format(time_pre, time_frame_pre))

    # CNN inference
    video_input = np.expand_dims(video_input, axis=-1)
    prob_map = fff.predict(video_input, batch_size=batch_size_eval)
    if display:
        end_network = time.time()
        time_CNN = end_network-end_pre
        time_frame_CNN = time_CNN/nframes*1000
        print('CNN Infrence time: {:6f} s, {:6f} ms/frame'.format(time_CNN, time_frame_CNN))

    # post-processing
    prob_map = prob_map.squeeze()[:, :Lx, :Ly]
    print(Params_post)
    Params_post_copy = Params_post.copy()
    Params_post_copy['thresh_pmap'] = None # Avoid repeated thresholding in postprocessing
    pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
    # threshold the probability map to binary activity
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)

    # the rest of post-processing. The result is a 2D sparse matrix of the segmented neurons
    Masks_2, times_active = complete_segment(pmaps_b, Params_post_copy, display=display, p=p, useWT=useWT)
    if display:
        finish = time.time()
        time_post = finish-end_network
        time_frame_post = time_post/nframes*1000
        print('Post-Processing time: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

    # convert to a 3D array of the segmented neurons
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')

    # Save total processing time, and average processing time per frame
    if display:
        time_all = finish-start
        time_frame_all = time_all/nframes*1000
        print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
        time_total = np.array([time_pre, time_CNN, time_post, time_all])
        time_frame = np.array([time_frame_pre, time_frame_CNN, time_frame_post, time_frame_all])
    else:
        time_total = np.zeros((4,))
        time_frame = np.zeros((4,))

    return Masks, Masks_2, times_active, time_total, time_frame#, pmaps_b


def suns_online(filename_video, filename_CNN, Params_pre, Params_post, dims, \
        frames_init, merge_every, batch_size_init=1, useSF=True, useTF=True, useSNR=True, \
        med_subtract=False, update_baseline=False, \
        useWT=False, show_intermediate=True, prealloc=True, display=True, useMP=True, p=None):
    '''The complete SUNS online procedure.
        It uses the trained CNN model from "filename_CNN" and the optimized hyper-parameters in "Params_post"
        to process the video "Exp_ID" in "dir_video"

    Inputs: 
        filename_video (str): The path of the file of the input raw video.
            The file must be a ".h5" file, with dataset "mov" being the input video (shape = (T0,Lx0,Ly0)).
        filename_CNN (str): The path of the trained CNN model. 
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
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        show_intermediate (bool, default to True): Indicator of whether 
            consecutive frame requirement is applied to screen neurons after every update. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.
        display (bool, default to True): Indicator of whether to show intermediate information
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        Masks (3D numpy.ndarray of bool, shape = (n,Lx0,Ly0)): the final segmented masks. 
        Masks_2 (scipy.csr_matrix of bool, shape = (n,Lx0*Ly0)): the final segmented masks in the form of sparse matrix. 
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
    init_imgs = np.zeros((batch_size_init, rowspad, colspad, 1), dtype='float32')
    init_masks = np.zeros((batch_size_init, rowspad, colspad, 1), dtype='uint8')
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
    # thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version


    # Spatial filtering preparation
    if useSF==True:
        # lateral dimensions slightly larger than the raw video but faster for FFT
        rows1 = cv2.getOptimalDFTSize(rowspad)
        cols1 = cv2.getOptimalDFTSize(colspad)
        
        # if the learned 2D and 3D wisdom files have been saved, load them. 
        # Otherwise, learn wisdom later
        Length_data2=str((rows1, cols1))
        cc2 = load_wisdom_txt(os.path.join('wisdom', Length_data2))
        
        Length_data3=str((frames_init, rows1, cols1))
        cc3 = load_wisdom_txt(os.path.join('wisdom', Length_data3))
        if cc3:
            pyfftw.import_wisdom(cc3)

        # mask for spatial filter
        mask2 = plan_mask2(dims, (rows1, cols1), gauss_filt_size)
        # FFT planning
        (bb, bf, fft_object_b, fft_object_c) = plan_fft(frames_init, (rows1, cols1), prealloc)
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
        if update_baseline:
            video_tf_past = np.ones((frames_init, rowspad, colspad), dtype='float32')        
    else:
        med_frame2 = np.zeros((rowspad, colspad, 2), dtype='float32')
        video_input = np.zeros((frames_initf, rowspad, colspad), dtype='float32')        
        pmaps_b_init = np.zeros((frames_initf, Lx, Ly), dtype='uint8')        
        frame_SNR = np.zeros(dimspad, dtype='float32')
        pmaps_b = np.zeros(dims, dtype='uint8')
        if update_baseline:
            video_tf_past = np.zeros((frames_init, rowspad, colspad), dtype='float32')        

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
    med_frame3, segs_all, recent_frames = init_online(
        bb, dims, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
        med_frame2, mask2, bf, fft_object_b, fft_object_c, Poisson_filt, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, \
        useWT=useWT, batch_size_init=batch_size_init, p=p)
    if useTF==True:
        past_frames[:leng_tf] = recent_frames
    tuple_temp = merge_complete(segs_all[:frames_initf], dims, Params_post)
    if show_intermediate:
        Masks_2 = select_cons(tuple_temp)

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
        (bb, bf, fft_object_b, fft_object_c) = plan_fft2((rows1, cols1))
    else:
        (bf, fft_object_b, fft_object_c) = (None, None, None)
        bb=np.zeros(dimspad, dtype='float32')
    
    print('Start frame by frame processing')
    # %% Online processing for the following frames
    current_frame = leng_tf+1
    t_merge = frames_initf
    for t in range(frames_initf,nframesf):
        if display:
            start_frame = time.time()
        # load the current frame
        bb[:Lx, :Ly] = video_raw[t+leng_tf-1]
        bb[Lx:, :] = 0
        bb[:, Ly:] = 0
        
        # PreProcessing
        frame_SNR, frame_tf = preprocess_online(bb, dimspad, med_frame3, frame_SNR, \
            past_frames[current_frame-leng_tf:current_frame], mask2, bf, fft_object_b, \
            fft_object_c, Poisson_filt, useSF=useSF, useTF=useTF, useSNR=useSNR, \
            med_subtract=med_subtract, update_baseline=update_baseline)

        if update_baseline:
            t_past = (t-frames_initf) % frames_init
            video_tf_past[t_past] = frame_tf
            if t_past == frames_init-1: 
            # update median and median-based standard deviation every "frames_init" frames
                if useSNR:
                    med_frame3 = SNR_normalization(
                        video_tf_past, med_frame2, (rowspad, colspad), 1, display=False)
                else:
                    med_frame3 = median_normalization(
                        video_tf_past, med_frame2, (rowspad, colspad), 1, display=False)

        # CNN inference
        frame_prob = CNN_online(frame_SNR, fff, dims)

        # first step of post-processing
        segs = separate_neuron_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)
        segs_all.append(segs)

        # temporal merging 1: combine neurons with COM distance smaller than thresh_COM0
        if ((t + 1 - t_merge) == merge_every) or (t==nframesf-1):
            # uniques, times_uniques = unique_neurons1_simp(segs_all[t_merge:], thresh_COM0) # minArea,
            totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs_all[t_merge:])
            uniques, times_uniques = unique_neurons2_simp(totalmasks, neuronstate, COMs, \
                areas, probmapID, minArea=0, thresh_COM0=thresh_COM0, useMP=useMP)

        # temporal merging 2: combine neurons with COM distance smaller than thresh_COM
        if ((t - 0 - t_merge) == merge_every) or (t==nframesf-1):
            if uniques.size:
                groupedneurons, times_groupedneurons = \
                    group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques, useMP=useMP)

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
                    have_cons_add = refine_seperate_cons_online(times_add, cons)
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
            tuple_temp = merge_2(tuple_temp, tuple_add, dims, Params_post)
            t_merge += merge_every
            if show_intermediate:
                Masks_2 = select_cons(tuple_temp)

        current_frame +=1
        # Update the stored latest frames when it runs out: move them "leng_tf" ahead
        if current_frame > leng_past:
            current_frame = leng_tf+1
            past_frames[:leng_tf] = past_frames[-leng_tf:]
        if display:
            end_frame = time.time()
            list_time_per[t] = end_frame - start_frame
        if t % 1000 == 0:
            print('{} frames has been processed'.format(t))

    if not show_intermediate:
        Masks_2 = select_cons(tuple_temp)
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
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
    return Masks, Masks_2, time_total, time_frame


def suns_online_track(filename_video, filename_CNN, Params_pre, Params_post, dims, \
        frames_init, merge_every, batch_size_init=1, useSF=True, useTF=True, useSNR=True, \
        med_subtract=False, update_baseline=False, \
        useWT=False, prealloc=True, display=True, useMP=True, p=None):
    '''The complete SUNS online procedure with tracking.
        It uses the trained CNN model from "filename_CNN" and the optimized hyper-parameters in "Params_post"
        to process the video "Exp_ID" in "dir_video"

    Inputs: 
        filename_video (str): The path of the file of the input raw video.
            The file must be a ".h5" file, with dataset "mov" being the input video (shape = (T0,Lx0,Ly0)).
        filename_CNN (str): The path of the trained CNN model. 
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
        med_subtract (bool, default to False): True if the spatial median of every frame is subtracted before temporal filtering.
            Can only be used when spatial filtering is not used. 
        update_baseline (bool, default to False): True if the median and median-based std is updated every "frames_init" frames.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.
        display (bool, default to True): Indicator of whether to show intermediate information
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        Masks (3D numpy.ndarray of bool, shape = (n,Lx0,Ly0)): the final segmented masks. 
        Masks_2 (scipy.csr_matrix of bool, shape = (n,Lx0*Ly0)): the final segmented masks in the form of sparse matrix. 
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
    init_imgs = np.zeros((batch_size_init, rowspad, colspad, 1), dtype='float32')
    init_masks = np.zeros((batch_size_init, rowspad, colspad, 1), dtype='uint8')
    fff.evaluate(init_imgs, init_masks, batch_size=batch_size_init)
    del init_imgs, init_masks

    # load optimal post-processing parameters
    minArea = Params_post['minArea']
    avgArea = Params_post['avgArea']
    # thresh_pmap = Params_post['thresh_pmap']
    thresh_mask = Params_post['thresh_mask']
    # thresh_COM0 = Params_post['thresh_COM0']
    # thresh_COM = Params_post['thresh_COM']
    thresh_IOU = Params_post['thresh_IOU']
    thresh_consume = Params_post['thresh_consume']
    # cons = Params_post['cons']
    # thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version


    # Spatial filtering preparation
    if useSF==True:
        # lateral dimensions slightly larger than the raw video but faster for FFT
        rows1 = cv2.getOptimalDFTSize(rowspad)
        cols1 = cv2.getOptimalDFTSize(colspad)
        
        # if the learned 2D and 3D wisdom files have been saved, load them. 
        # Otherwise, learn wisdom later
        Length_data2=str((rows1, cols1))
        cc2 = load_wisdom_txt(os.path.join('wisdom', Length_data2))
        
        Length_data3=str((frames_init, rows1, cols1))
        cc3 = load_wisdom_txt(os.path.join('wisdom', Length_data3))
        if cc3:
            pyfftw.import_wisdom(cc3)

        # mask for spatial filter
        mask2 = plan_mask2(dims, (rows1, cols1), gauss_filt_size)
        # FFT planning
        (bb, bf, fft_object_b, fft_object_c) = plan_fft(frames_init, (rows1, cols1), prealloc)
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
        if update_baseline:
            video_tf_past = np.ones((frames_init, rowspad, colspad), dtype='float32')        
    else:
        med_frame2 = np.zeros((rowspad, colspad, 2), dtype='float32')
        video_input = np.zeros((frames_initf, rowspad, colspad), dtype='float32')        
        pmaps_b_init = np.zeros((frames_initf, Lx, Ly), dtype='uint8')        
        frame_SNR = np.zeros(dimspad, dtype='float32')
        pmaps_b = np.zeros(dims, dtype='uint8')
        if update_baseline:
            video_tf_past = np.zeros((frames_init, rowspad, colspad), dtype='float32')        

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
    med_frame3, segs_all, recent_frames = init_online(
        bb, dims, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
        med_frame2, mask2, bf, fft_object_b, fft_object_c, Poisson_filt, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, \
        useWT=useWT, batch_size_init=batch_size_init, p=p)
    if useTF==True:
        past_frames[:leng_tf] = recent_frames
    tuple_temp = merge_complete(segs_all[:frames_initf], dims, Params_post)

    # Initialize Online track variables
    (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp) = tuple_temp
    # list of previously found neurons that satisfy consecutive frame requirement
    Masks_cons = select_cons(tuple_temp) 
    # sparse matrix of previously found neurons that satisfy consecutive frame requirement
    Masks_cons_2D = sparse.vstack(Masks_cons) 
    # indices of previously found neurons that satisfy consecutive frame requirement
    ind_cons = have_cons_temp.nonzero()[0]
    segs0 = segs_all[0] # segs of initialization frames
    # segs if no neurons are found
    segs_empty = (segs0[0][0:0], segs0[1][0:0], segs0[2][0:0], segs0[3][0:0]) 
    # Number of previously found neurons that satisfy consecutive frame requirement
    N1 = len(Masks_cons)
    # list of "segs" for neurons that are not previously found
    list_segs_new = [] 
    # list of newly segmented masks for old neurons (segmented in previous frames)
    list_masks_old = [[] for _ in range(N1)] 
    # list of the newly active indices of frames of old neurons
    times_active_old = [[] for _ in range(N1)] 
    # True if the old neurons are active in the previous frame
    active_old_previous = np.zeros(N1, dtype='bool')

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
        (bb, bf, fft_object_b, fft_object_c) = plan_fft2((rows1, cols1))
    else:
        (bf, fft_object_b, fft_object_c) = (None, None, None)
        bb=np.zeros(dimspad, dtype='float32')

    print('Start frame by frame processing')
    # %% Online processing for the following frames
    current_frame = leng_tf+1
    t_merge = frames_initf
    for t in range(frames_initf, nframesf):
        if display:
            start_frame = time.time()
        # load the current frame
        bb[:Lx, :Ly] = video_raw[t+leng_tf-1]
        bb[Lx:, :] = 0
        bb[:, Ly:] = 0

        # PreProcessing
        frame_SNR, frame_tf = preprocess_online(bb, dimspad, med_frame3, frame_SNR, \
            past_frames[current_frame-leng_tf:current_frame], mask2, bf, fft_object_b, fft_object_c, \
            Poisson_filt, useSF=useSF, useTF=useTF, useSNR=useSNR, \
            med_subtract=med_subtract, update_baseline=update_baseline)

        if update_baseline:
            t_past = (t-frames_initf) % frames_init
            video_tf_past[t_past] = frame_tf
            if t_past == frames_init-1: 
            # update median and median-based standard deviation every "frames_init" frames
                if useSNR:
                    med_frame3 = SNR_normalization(
                        video_tf_past, med_frame2, (rowspad, colspad), 1, display=False)
                else:
                    med_frame3 = median_normalization(
                        video_tf_past, med_frame2, (rowspad, colspad), 1, display=False)

        # CNN inference
        frame_prob = CNN_online(frame_SNR, fff, dims)

        # first step of post-processing
        segs = separate_neuron_online(frame_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT)

        active_old = np.zeros(N1, dtype='bool') # True if the old neurons are active in the current frame
        masks_t, neuronstate_t, cents_t, areas_t = segs
        N2 = neuronstate_t.size
        if N2: # Try to merge the new masks to old neurons
            new_found = np.zeros(N2, dtype='bool')
            for n2 in range(N2):
                masks_t2 = masks_t[n2]
                cents_t2 = int(np.round(cents_t[n2,1]) * Ly + np.round(cents_t[n2,0]))  
                # If a new masks belongs to an old neuron, the COM of the new mask must be inside the old neuron area.
                # Select possible old neurons that the new mask can merge to
                possible_masks1 = Masks_cons_2D[:,cents_t2].nonzero()[0]
                IOUs = np.zeros(len(possible_masks1))
                areas_t2 = areas_t[n2]
                for (ind,n1) in enumerate(possible_masks1):
                    # Calculate IoU and consume ratio to determine merged neurons
                    area_i = Masks_cons[n1].multiply(masks_t2).nnz
                    area_temp1 = area_temp[n1]
                    area_u = area_temp1 + areas_t2 - area_i
                    IOU = area_i / area_u
                    consume = area_i / min(area_temp1, areas_t2)
                    contain = (IOU >= thresh_IOU) or (consume >= thresh_consume)
                    if contain: # merging criterion satisfied
                        IOUs[ind] = IOU
                num_contains = IOUs.nonzero()[0].size
                if num_contains: # The new mask can merge to one of the old neurons.
                    # If there are multiple candicates, choose the one with the highest IoU
                    belongs = possible_masks1[IOUs.argmax()]
                    # merge the mask and active frame index
                    list_masks_old[belongs].append(masks_t2)
                    times_active_old[belongs].append(t + frames_initf)
                    # This old neurons is active in the current frame
                    active_old[belongs] = True 
                else: # The new mask can not merge to any old neuron.
                    new_found[n2] = True

            if np.any(new_found): # There are some new masks that can not merge to old neurons
                segs_new = (masks_t[new_found], neuronstate_t[new_found], cents_t[new_found], areas_t[new_found])
            else: # All masks already merged to old neurons
                segs_new = segs_empty
                
        else: # No neurons fould in the current frame
            segs_new = segs
        list_segs_new.append(segs_new)

        if (t + 1 - t_merge) != merge_every or t == (nframesf-1):
            # Update the old neurons with new appearances in the current frame.
            if t < (nframesf-1):
                # True if the neurons are active in the previous frame but not active in the current frame
                inactive = np.logical_and(active_old_previous, np.logical_not(active_old)).nonzero()[0]
            else: # last frame
                # All active neurons should be updated, so they are treated as inactive in the next frame
                inactive = active_old_previous.nonzero()[0]

            # Update the indicators of the previous frame using the current frame
            active_old_previous = active_old.copy()
            for n1 in inactive: 
                # merge new active frames to existing active frames for already found neurons
                # n1 is the index in the old neurons that satisfy consecutive frame requirement. 
                # n10 is the index in all old neurons.
                n10 = ind_cons[n1] 
                # Add all the new masks to the overall real-number masks
                mask_update = masks_temp[n10] + sum(list_masks_old[n1])
                masks_temp[n10] = mask_update
                # Add indices of active frames
                times_add = np.unique(np.array(times_active_old[n1]))
                times_temp[n10] = np.hstack([times_temp[n10], times_add])
                # reset lists used to store the information from new frames related to old neurons
                list_masks_old[n1] = []
                times_active_old[n1] = []
                # update the binary masks and areas
                Maskb_update = mask_update >= mask_update.max() * thresh_mask
                Masksb_temp[n10] = Maskb_update
                Masks_cons[n1] = Maskb_update
                area_temp[n10] = Maskb_update.nnz
            if inactive.size:
                Masks_cons_2D = sparse.vstack(Masks_cons) 

        if (t + 1 - t_merge) == merge_every or t == (nframesf-1):
            if t < (nframesf-1):
                # delay merging new frame to next frame by assuming all the neurons active in the previous frame
                # are still active in the current frame, to reserve merging time for new neurons
                active_old_previous = np.logical_or(active_old_previous, active_old)

            # merge new neurons with old masks that do not satisfy consecutive frame requirement
            tuple_temp = (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp)
            # merge the remaining new masks from the most recent "merge_every" frames
            tuple_add = merge_complete(list_segs_new, dims, Params_post)
            (Masksb_add, masks_add, times_add, area_add, have_cons_add) = tuple_add
            # update the indices of active frames
            times_add = [x + merge_every for x in times_add]
            tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)
            # merge the remaining new masks with the existing masks that do not satisfy consecutive frame requirement
            tuple_temp = merge_2_nocons(tuple_temp, tuple_add, dims, Params_post)

            (Masksb_temp, masks_temp, times_temp, area_temp, have_cons_temp) = tuple_temp
            # Update the indices of old neurons that satisfy consecutive frame requirement
            ind_cons_new = have_cons_temp.nonzero()[0]
            for (ind,ind_cons_0) in enumerate(ind_cons_new):
                if ind_cons_0 not in ind_cons: 
                    # update lists used to store the information from new frames related to old neurons
                    if ind_cons_0 > ind_cons.max():
                        list_masks_old.append([])
                        times_active_old.append([])
                    else:
                        list_masks_old.insert(ind, [])
                        times_active_old.insert(ind, [])
            
            # Update the list of previously found neurons that satisfy consecutive frame requirement
            Masks_cons = select_cons(tuple_temp)
            Masks_cons_2D = sparse.vstack(Masks_cons) 
            N1 = len(Masks_cons)
            list_segs_new = []
            # Update whether the old neurons are active in the previous frame
            active_old_previous = np.zeros_like(have_cons_temp)
            active_old_previous[ind_cons] = active_old
            active_old_previous = active_old_previous[ind_cons_new]
            ind_cons = ind_cons_new
            t_merge += merge_every

        current_frame +=1
        # Update the stored latest frames when it runs out: move them "leng_tf" ahead
        if current_frame > leng_past:
            current_frame = leng_tf+1
            past_frames[:leng_tf] = past_frames[-leng_tf:]
        if display:
            end_frame = time.time()
            list_time_per[t] = end_frame - start_frame
        if t % 1000 == 0:
            print('{} frames has been processed'.format(t))

    Masks_cons = select_cons(tuple_temp)
    # final result. Masks_2 is a 2D sparse matrix of the segmented neurons
    if len(Masks_cons):
        Masks_2 = sparse.vstack(Masks_cons)
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
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
    return Masks, Masks_2, time_total, time_frame
