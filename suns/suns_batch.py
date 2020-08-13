# %%
import os
import numpy as np
import time
import sys

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\PreProcessing')
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

import par1
from preprocessing_functions import preprocess_video
from par3 import fastthreshold
from shallow_unet import get_shallow_unet
from complete_post import complete_segment


def suns_batch(dir_video, Exp_ID, filename_CNN, Params_pre, Params_post, dims, \
        batch_size_eval=1, useSF=True, useTF=True, useSNR=True, \
        useWT=False, prealloc=True, display=True, useMP=True, p=None):
    '''The complete SUNS batch procedure.

    Inputs: 
        dir_video (str): The folder containing the input video.
        Exp_ID (str): The filer name of the input video. 
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
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useSF (bool, default to True): True if spatial filtering is used.
        useTF (bool, default to True): True if temporal filtering is used.
        useSNR (bool, default to True): True if pixel-by-pixel SNR normalization filtering is used.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        prealloc (bool, default to True): True if pre-allocate memory space for large variables. 
            Achieve faster speed at the cost of higher memory occupation.
        display (bool, default to True): Indicator of whether to show intermediate information
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        Masks (3D numpy.ndarray of bool, shape = (n,Lx,Ly)): the final segmented masks. 
        Masks_2 (scipy.csr_matrix of bool, shape = (n,Lx*Ly)): the final segmented masks in the form of sparse matrix. 
        time_total (list of float, shape = (4,)): the total time spent 
            for pre-processing, CNN inference, post-processing, and total processing
        time_frame (list of float, shape = (4,)): the average time spent on every frame
            for pre-processing, CNN inference, post-processing, and total processing
    '''
    if display:
        start = time.time()
    (Lx, Ly) = dims
    # load CNN model
    fff = get_shallow_unet()
    fff.load_weights(filename_CNN)
    # run CNN inference once to warm up
    init_imgs = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='float32')
    init_masks = np.zeros((batch_size_eval, Lx, Ly, 1), dtype='uint8')
    fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)
    del init_imgs, init_masks

    thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    # thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version
    if display:
        time_init = time.time()
        print('Initialization time: {} s'.format(time_init-start))

    # %% Actual processing starts after the video is loaded into memory
        # which is in the middle of "preprocess_video", represented by the output "start"
    # pre-processing including loading data
    video_input, start = preprocess_video(dir_video, Exp_ID, Params_pre, \
        useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc, display=display)
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
    Params_post['thresh_pmap'] = None # Avoid repeated thresholding in postprocessing
    pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
    # threshold the probability map to binary activity
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)

    # the rest of post-processing. The result is a 2D sparse matrix of the segmented neurons
    Masks_2 = complete_segment(pmaps_b, Params_post, display=display, p=p, useWT=useWT)
    if display:
        finish = time.time()
        time_post = finish-end_network
        time_frame_post = time_post/nframes*1000
        print('Post-Processing time: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

    # convert to a 3D array of the segmented neurons
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly))

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

    return Masks, Masks_2, time_total, time_frame


