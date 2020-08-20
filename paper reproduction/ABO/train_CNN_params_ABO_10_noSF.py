# %%
import sys
import os
import random
import time
import glob
import numpy as np
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video
from suns.PreProcessing.generate_masks import generate_masks
from suns.train_CNN_params import train_CNN, parameter_optimization_cross_validation


# %%
if __name__ == '__main__':
    # %% setting parameters
    rate_hz = 30 # frame rate of the video
    Dimens = (487,487) # lateral dimensions of the video
    nframes = 23200 # number of frames for each video
    Mag = 1 # spatial magnification compared to ABO videos.

    thred_std = 6 # SNR threshold used to determine when neurons are active
    num_train_per = 200 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 200 # batch size in CNN inference

    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.
    useWT=False # True if using additional watershed
    load_exist=False # True if using temp files already saved in the folders
    use_validation = False # True to use a validation set outside the training set
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "use_all"
    Params_loss = {'DL':1, 'BCE':0, 'FL':100, 'gamma':1, 'alpha':0.25} # Parameters of the loss function

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    # folder of the raw videos
    dir_video = 'D:\\ABO\\20 percent\\' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'Markings\\Layer275\\FinalGT\\FinalMasks_' 
    dir_parent = dir_video + 'noSF\\' # folder to save all the processed data
    dir_network_input = dir_parent + 'network_input\\' # folder of the SNR videos
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks
    weights_path = dir_parent + 'Weights\\' # folder to save the trained CNN
    training_output_path = dir_parent + 'training output\\' # folder to save the loss functions during training
    dir_output = dir_parent + 'output_masks\\' # folder to save the optimized hyper-parameters
    dir_temp = dir_parent + 'temp\\' # temporary folder to save the F1 with various hyper-parameters

    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    (rows, cols) = Dimens # size of the network input and output
    (Lx, Ly) = (rows, cols) # size of the original video
    num_total = nframes # number of frames of the video

    # %% set pre-processing parameters
    nn = nframes
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.
    filename_TF_template = 'GCaMP6f_spike_tempolate_mean.h5'

    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'nn':nn, 'Poisson_filt': Poisson_filt}

    # %% set the range of post-processing hyper-parameters to be optimized in
    # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
    list_minArea = list(range(80,135,10)) 
    # average area of a typical neuron (unit: pixels in ABO videos)
    list_avgArea = [177] 
    # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
    list_thresh_pmap = list(range(165,210,5))
    # threshold to binarize the neuron masks. For each mask, 
    # values higher than "thresh_mask" times the maximum value of the mask are set to one.
    thresh_mask = 0.5
    # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
    thresh_COM0 = 2
    # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
    list_thresh_COM = list(np.arange(4, 9, 1)) 
    # minimum IoU of two masks to be considered the same neuron
    list_thresh_IOU = [0.5] 
    # minimum consecutive number of frames of active neurons
    list_cons = list(range(1, 8, 1)) 

    # adjust the units of the hyper-parameters to pixels in the test videos according to relative magnification
    list_minArea= list(np.round(np.array(list_minArea) * Mag**2))
    list_avgArea= list(np.round(np.array(list_avgArea) * Mag**2))
    thresh_COM0= thresh_COM0 * Mag
    list_thresh_COM= list(np.array(list_thresh_COM) * Mag)
    # adjust the minimum consecutive number of frames according to different frames rates between ABO videos and the test videos
    # list_cons=list(np.round(np.array(list_cons) * rate_hz/30).astype('int'))

    # dictionary of all fixed and searched post-processing parameters.
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
            'thresh_mask': thresh_mask, 'list_cons': list_cons}
    print(Params_set)


    # # pre-processing for training
    # for Exp_ID in list_Exp_ID: #
    #     # %% Pre-process video
    #     video_input, _ = preprocess_video(dir_video, Exp_ID, Params, dir_network_input, \
    #         useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) #

    #     # %% Determine active neurons in all frames using FISSA
    #     file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
    #     generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
    #     del video_input

    # %% CNN training
    for CV in [10]:
        list_Exp_ID_train = list_Exp_ID.copy()
        list_Exp_ID_val = None # get rid of validation steps
        file_CNN = weights_path+'Model_CV{}.h5'.format(CV)
        results = train_CNN(dir_network_input, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
            BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rows, cols), Params_loss)

        # save training and validation loss after each eopch
        f = h5py.File(training_output_path+"training_output_CV{}.h5".format(CV), "w")
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        if use_validation:
            f.create_dataset("val_loss", data=results.history['val_loss'])
            f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.close()

    # %% parameter optimization
    parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        (Lx, Ly), (rows, cols), dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
        batch_size_eval, useWT=useWT, useMP=True, load_exist=load_exist)
