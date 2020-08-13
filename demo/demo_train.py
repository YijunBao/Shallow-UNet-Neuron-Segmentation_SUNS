# %%
import sys
import os
import random
import numpy as np
import tensorflow as tf
import h5py

sys.path.insert(1, '..\\Network')
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from data_gen import data_gen
from shallow_unet import get_shallow_unet
import time


def train_CNN(dir_img, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
    BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, dims):
    nvideo_train = len(list_Exp_ID_train)
    nvideo_val = len(list_Exp_ID_val)
    (rows, cols) = dims

    train_every = num_total//num_train_per
    start_frame_train = random.randint(0,train_every-1)
    NO_OF_TRAINING_IMAGES = num_train_per * nvideo_train
    # set how to choose validation images
    num_val_per = int((num_train_per * nvideo_train / nvideo_val) // 9)
    num_val_per = min(num_val_per, num_total)
    val_every = num_total//num_val_per
    start_frame_val = random.randint(0,val_every-1)
    NO_OF_VAL_IMAGES = num_val_per * nvideo_val

    # %% Load traiming images and masks from h5 files
    val_imgs = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='float32') # validation images
    val_masks = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='uint8') # temporal masks for validation images
    train_imgs = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='float32') # training images
    train_masks = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='uint8') # temporal masks for training images

    print('Loading training images and masks.')
    # Select validation images: start from frame "start_frame", 
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
        train_imgs[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
            = np.array(h5_img['network_input'][start_frame_train:train_every*num_train_per:train_every])
        h5_img.close()
        h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        train_masks[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
            = np.array(h5_mask['temporal_masks'][start_frame_train:train_every*num_train_per:train_every])
        h5_mask.close()

    # generater for training and validation images and masks
    train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
    val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


    fff = get_shallow_unet()

    class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))
    # train CNN
    results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=[LossAndErrorPrintingCallback()])

    # save trained CNN model 
    fff.save_weights(file_CNN)
    return results


# %%
if __name__ == '__main__':
    # %% Setting parameters
    Dimens = (120,88) # lateral dimensions of the video
    Nframes = 3000 # number of frames of the video
    thred_std = 3 # SNR threshold used to determine when neurons are active
    num_train_per = 2400 # Number of frames per video used for training 
    BATCH_SIZE = 20 # Batch size for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 

    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']
    # folder of the raw videos
    dir_video = 'data\\'
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "leave_one_out"

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    # if cross_validation == "leave_one_out":
    #     nvideo_train = nvideo-1
    #     nvideo_val = 1
    # else: # cross_validation == "train_1_test_rest"
    #     nvideo_train = 1
    #     nvideo_val = nvideo-1

    dir_parent = dir_video + 'complete\\' # folder to save all the processed data
    dir_sub = ''
    dir_img = dir_parent + 'network_input\\' # folder to save the SNR videos
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks
    weights_path = dir_parent+dir_sub + 'Weights\\' # folder to save the trained CNN
    training_output_path = dir_parent+dir_sub + 'training output\\' # folder to save the loss functions during training
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 

    (Lx, Ly) = (rows, cols) = Dimens # lateral dimensions of the video
    num_total = Nframes # number of frames of the video
    # # set how to choose training images
    # train_every = num_total//num_train_per
    # start_frame_train = random.randint(0,train_every-1)
    # NO_OF_TRAINING_IMAGES = num_train_per * nvideo_train
    # # set how to choose validation images
    # num_val_per = int((num_train_per * nvideo_train / nvideo_val) // 9)
    # num_val_per = min(num_val_per, num_total)
    # val_every = num_total//num_val_per
    # start_frame_val = random.randint(0,val_every-1)
    # NO_OF_VAL_IMAGES = num_val_per * nvideo_val
        

    # %% start training
    for CV in range(0,nvideo):
        if cross_validation == "leave_one_out":
            list_Exp_ID_train = list_Exp_ID.copy()
            list_Exp_ID_val = [list_Exp_ID_train.pop(CV)]
        else: # cross_validation == "train_1_test_rest"
            list_Exp_ID_val = list_Exp_ID.copy()
            list_Exp_ID_train = [list_Exp_ID_val.pop(CV)]
        file_CNN = weights_path+'Model_CV{}.h5'.format(CV)
        results = train_CNN(dir_img, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
            BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, (rows, cols))

        # # %% Load traiming images and masks from h5 files
        # val_imgs = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='float32') # validation images
        # val_masks = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='uint8') # temporal masks for validation images
        # train_imgs = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='float32') # training images
        # train_masks = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='uint8') # temporal masks for training images

        # print('Loading training images and masks.')
        # # Select validation images: start from frame "start_frame", 
        # # select a frame every "val_every" frames, totally "num_val_per" frames  
        # for cnt, Exp_ID in enumerate(list_Exp_ID_val):
        #     h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        #     val_imgs[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
        #         = np.array(h5_img['network_input'][start_frame_val:val_every*num_val_per:val_every])
        #     h5_img.close()
        #     h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        #     val_masks[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
        #         = np.array(h5_mask['temporal_masks'][start_frame_val:val_every*num_val_per:val_every])
        #     h5_mask.close()

        # # h5_img = h5py.File(dir_img+Exp_ID_val+'.h5', 'r')
        # # (nframes, Lx, Ly) = h5_img['network_input'].shape
        # # if nframes<num_val_per:
        # #     num_val_per=nframes
        # #     val_imgs = np.zeros((num_val_per, rows, cols), dtype='float32')
        # #     val_masks = np.zeros((num_val_per, rows, cols), dtype='uint8')
        # # # Select validation images: start from frame "start_frame", 
        # # # select a frame every "val_every" frames, totally "num_val_per" frames  
        # # val_imgs[:,:Lx,:Ly] = np.array(h5_img['network_input'][start_frame_train:val_every*num_val_per:val_every])
        # # h5_img.close()
        # # h5_mask = h5py.File(dir_mask+Exp_ID_val+'.h5', 'r')
        # # val_masks[:,:Lx,:Ly] = np.array(h5_mask['temporal_masks'][start_frame_train:val_every*num_val_per:val_every])
        # # h5_mask.close()

        # # Select training images: for each video, start from frame "start_frame", 
        # # select a frame every "train_every" frames, totally "train_val_per" frames  
        # for cnt, Exp_ID in enumerate(list_Exp_ID_train):
        #     h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        #     train_imgs[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
        #         = np.array(h5_img['network_input'][start_frame_train:train_every*num_train_per:train_every])
        #     h5_img.close()
        #     h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        #     train_masks[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
        #         = np.array(h5_mask['temporal_masks'][start_frame_train:train_every*num_train_per:train_every])
        #     h5_mask.close()

        # # generater for training and validation images and masks
        # train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
        # val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


        # fff = get_shallow_unet()

        # class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        #     def on_epoch_end(self, epoch, logs=None):
        #         print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))
        # # train CNN
        # results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
        #                         validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=[LossAndErrorPrintingCallback()])

        # # save trained CNN model 
        # fff.save_weights(weights_path+'Model_CV{}.h5'.format(CV))

        # save training and validation loss after each eopch
        f = h5py.File(training_output_path+"training_output_CV{}.h5".format(CV), "w")
        f.create_dataset("val_loss", data=results.history['val_loss'])
        f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        f.close()

