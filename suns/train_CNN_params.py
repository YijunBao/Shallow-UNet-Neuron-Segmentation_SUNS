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
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.Network.data_gen import data_gen
from suns.Network.shallow_unet import get_shallow_unet
from suns.Network.par2 import fastuint
from suns.PostProcessing.complete_post import parameter_optimization


def train_CNN(dir_img, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
    BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, dims):
    '''Train a CNN model.

    Inputs: 
        dir_img (str): The folder containing the network_input.
        dir_mask (str): The folder containing the temporal masks.
        file_CNN (str): The path to save the trained CNN model.
        list_Exp_ID_train (list of str): The list of file names of the training video(s). 
        list_Exp_ID_train (list of str): The list of file names of the validation video(s). 
        BATCH_SIZE (int): batch size for CNN training.
        NO_OF_EPOCHS (int): number of epochs for CNN training.
        num_train_per (int): number of training images per video.
        num_total (int): total number of frames of a video (can be smaller than acutal number).
        dims (tuplel of int, shape = (2,)): lateral dimension of the video.

    Outputs:
        results: the training results containing the loss information.
        In addition, the trained CNN model is saved in "file_CNN".
    '''
    nvideo_train = len(list_Exp_ID_train) # Number of training videos
    nvideo_val = len(list_Exp_ID_val) # Number of validation videos
    (rows, cols) = dims

    # set how to choose training images
    train_every = num_total//num_train_per
    start_frame_train = random.randint(0,train_every-1)
    NO_OF_TRAINING_IMAGES = num_train_per * nvideo_train
    # set how to choose validation images
    # the total number of validation images is about 1/9 of the traning images
    num_val_per = int((num_train_per * nvideo_train / nvideo_val) // 9) 
    num_val_per = min(num_val_per, num_total)
    val_every = num_total//num_val_per
    start_frame_val = random.randint(0,val_every-1)
    NO_OF_VAL_IMAGES = num_val_per * nvideo_val

    # %% Load traiming images and masks from h5 files
    # validation images
    val_imgs = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='float32') 
    # temporal masks for validation images
    val_masks = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='uint8') 
    # training images
    train_imgs = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='float32') 
    # temporal masks for training images
    train_masks = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='uint8') 

    print('Loading training images and masks.')
    # Select validation images: for each video, start from frame "start_frame", 
    # select a frame every "val_every" frames, totally "num_val_per" frames  
    for cnt, Exp_ID in enumerate(list_Exp_ID_val):
        h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        val_imgs[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
            = np.array(h5_img['network_input'][start_frame_val:val_every*num_val_per:val_every])
        h5_img.close()
        h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        val_masks[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
            = np.array(h5_mask['temporal_masks'][start_frame_val:val_every*num_val_per:val_every])
        h5_mask.close()

    # Select training images: for each video, start from frame "start_frame", 
    # select a frame every "train_every" frames, totally "train_val_per" frames  
    for cnt, Exp_ID in enumerate(list_Exp_ID_train):
        h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        train_imgs[cnt*num_train_per:(cnt+1)*num_train_per,:,:] \
            = np.array(h5_img['network_input'][start_frame_train:train_every*num_train_per:train_every])
        h5_img.close()
        h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        train_masks[cnt*num_train_per:(cnt+1)*num_train_per,:,:] \
            = np.array(h5_mask['temporal_masks'][start_frame_train:train_every*num_train_per:train_every])
        h5_mask.close()

    # generater for training and validation images and masks
    train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
    val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


    fff = get_shallow_unet()
    # The alternative line has more options to choose
    # fff = get_shallow_unet_more(size=None, n_depth=3, n_channel=4, skip=[1], activation='elu')

    class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))
    # train CNN
    results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=[LossAndErrorPrintingCallback()])

    # save trained CNN model 
    fff.save_weights(file_CNN)
    return results


def parameter_optimization_pipeline(file_CNN, network_input, dims, \
        Params_set, filename_GT, batch_size_eval=1, useWT=False, useMP=True, p=None):
    '''The complete parameter optimization pipeline for one video and one CNN model.

    Inputs: 
        file_CNN (str): The path to save the trained CNN model.
        network_input (3D numpy.ndarray of float32): the SNR video obtained after pre-processing.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw video.
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float or int) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        list_Recall (6D numpy.array of float): Recall for all paramter combinations. 
        list_Precision (6D numpy.array of float): Precision for all paramter combinations. 
        list_F1 (6D numpy.array of float): F1 for all paramter combinations. 
            For these outputs, the orders of the tunable parameters are:
            "minArea", "avgArea", "thresh_pmap", "thresh_COM", "thresh_IOU", "cons"
    '''
    (Lx, Ly) = dims
    # load CNN model
    fff = get_shallow_unet()
    fff.load_weights(file_CNN)

    # CNN inference
    start_test = time.time()
    prob_map = fff.predict(network_input, batch_size=batch_size_eval)
    finish_test = time.time()
    Time_frame = (finish_test-start_test)/network_input.shape[0]*1000
    print('Average infrence time {} ms/frame'.format(Time_frame))

    # convert the output probability map from float to uint8 to speed up future parameter optimization
    prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
    pmaps = np.zeros(prob_map.shape, dtype='uint8')
    fastuint(prob_map, pmaps)
    del prob_map, fff

    # calculate the recall, precision, and F1 when different post-processing hyper-parameters are used.
    list_Recall, list_Precision, list_F1 = parameter_optimization(pmaps, Params_set, filename_GT, useWT=useWT, p=p)
    return list_Recall, list_Precision, list_F1


def parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        dims, dims1, dir_img, weights_path, dir_GTMasks, dir_temp, dir_output, \
            batch_size_eval=1, useWT=False, useMP=True, load_exist=False):
    '''The parameter optimization for a complete cross validation.

    Inputs: 
        cross_validation (str, can be "leave-one-out" or "train_1_test_rest"): 
            Represent the cross validation is either "leave-one-out" cross validation or "train 1 test rest".
        list_Exp_ID (list of str): The list of file names of all the videos. 
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float or int) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw video.
        dims1 (tuplel of int, shape = (2,)): lateral dimension of the padded video.
        dir_img (str): The path containing the SNR video after pre-processing.
        weights_path (str): The path containing the trained CNN model.
        dir_GTMasks (str): The path containing the GT masks.
        dir_temp (str): The path to save the recall, precision, and F1 of various parameters.
        dir_output (str): The path to save the optimal parameters.
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        load_exist (bool, default to False): Indicator of whether previous F1 of various parameters are loaded. 

    Outputs:
        No output variable, but the recall, precision, and F1 of various parameters are saved in folder "dir_temp"
            and the optimal parameters are saved in folder "dir_output"
    '''
    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    if cross_validation == "leave_one_out":
        nvideo_train = nvideo-1
    else: # cross_validation == "train_1_test_rest"
        nvideo_train = 1
    (Lx, Ly) = dims
    (rows, cols) = dims1

    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    thresh_mask = Params_set['thresh_mask']
    list_cons = Params_set['list_cons']

    size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
    # arrays to save the recall, precision, and F1 when different post-processing hyper-parameters are used.
    F1_train = np.zeros(size_F1)
    Recall_train = np.zeros(size_F1)
    Precision_train = np.zeros(size_F1)
    (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
        =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
        # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

    # %% start parameter optimization for each video with various CNN models
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        list_saved_results = glob.glob(dir_temp+'Parameter Optimization * Exp{}.mat'.format(Exp_ID))
        p = mp.Pool(mp.cpu_count())
        if len(list_saved_results)<nvideo_train or not load_exist: 
            # load SNR videos as "network_input"
            network_input = 0
            print('Video '+Exp_ID)
            start = time.time()
            h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
            nframes = h5_img['network_input'].shape[0]
            network_input = np.zeros((nframes, rows, cols, 1), dtype='float32')
            for t in range(nframes):
                network_input[t, :,:,0] = np.array(h5_img['network_input'][t])
            h5_img.close()
            time_load = time.time()
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            print('Load data: {} s'.format(time_load-start))

        if cross_validation == "leave_one_out":
            list_CV = list(range(nvideo))
            list_CV.pop(eid)
        else: # cross_validation == "train_1_test_rest"
            list_CV = [eid]

        for CV in list_CV:
            mat_filename = dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID)
            if os.path.exists(mat_filename) and load_exist: 
                # if the temporary output file already exists, load it
                mdict = loadmat(mat_filename)
                Recall_train[CV,eid] = np.array(mdict['list_Recall'])
                Precision_train[CV,eid] = np.array(mdict['list_Precision'])
                F1_train[CV,eid] = np.array(mdict['list_F1'])
        
            else: # Calculate recall, precision, and F1 for various parameters
                start = time.time()
                file_CNN = weights_path+'Model_CV{}.h5'.format(CV)
                list_Recall, list_Precision, list_F1 = parameter_optimization_pipeline(
                    file_CNN, network_input, (Lx,Ly), Params_set, filename_GT, batch_size_eval, useWT=useWT, useMP=useMP, p=p)
                
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                Recall_train[CV,eid] = list_Recall
                Precision_train[CV,eid] = list_Precision
                F1_train[CV,eid] = list_F1
                # save recall, precision, and F1 in a temporary ".mat" file
                mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
                savemat(mat_filename, mdict) 

        p.close()
            
    # %% Find the optimal postprocessing parameters
    for CV in range(nvideo):
        # calculate the mean recall, precision, and F1 of all the training videos
        Recall_mean = Recall_train[CV].mean(axis=0)*nvideo/nvideo_train
        Precision_mean = Precision_train[CV].mean(axis=0)*nvideo/nvideo_train
        F1_mean = F1_train[CV].mean(axis=0)*nvideo/nvideo_train
        Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
            array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
        print('F1_max=', [x.max() for x in F1_train[CV]])

        # find the post-processing hyper-parameters to achieve the highest average F1 over the training videos
        ind = F1_mean.argmax()
        ind = np.unravel_index(ind,F1_mean.shape)
        minArea = list_minArea[ind[0]]
        avgArea = list_avgArea[ind[1]]
        thresh_pmap = list_thresh_pmap[ind[2]]
        thresh_COM = list_thresh_COM[ind[3]]
        thresh_IOU = list_thresh_IOU[ind[4]]
        thresh_consume = (1+thresh_IOU)/2
        cons = list_cons[ind[5]]
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        print('F1_mean=', F1_mean[ind])

        # save the optimal hyper-parameters a ".mat" file
        Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
            'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
        savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)

