# %%
import sys
import os
# from unet4_4layers import get_unet
import random
import numpy as np
import cv2
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
import h5py

sys.path.insert(1, '..\\Network')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from data_gen_NF import data_gen_rectangle
from data_gen import data_gen
from unet4_best import get_unet
import time

# %%
if __name__ == '__main__':
    # time.sleep(3000)
    # radius = 6
    # rate_hz = 10
    # decay_time = 0.75
    Dimens = (120,88)
    nframes = 3000
    # Mag = radius/8

    thred_std = 3
    num_train_per = 2400
    BATCH_SIZE = 10
    NO_OF_EPOCHS = 50
    useSF=True
    useTF=True
    useSNR=True

    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']
    dir_video = 'data\\'
    dir_GTMasks = dir_video + 'FinalMasks_'

    nvideo = len(list_Exp_ID)
    # size = 248
    (rows, cols) = Dimens
    num_total = nframes
    train_every = num_total//num_train_per
    start_frame = random.randint(0,train_every-1)
    # Train the model
    NO_OF_TRAINING_IMAGES = num_train_per*(nvideo-1)
    # NO_OF_VAL_IMAGES = num_train_per
    num_val_per = num_train_per//3
    val_every = num_total//num_val_per
    NO_OF_VAL_IMAGES = num_val_per

    dir_parent = dir_video + 'complete\\'
    dir_sub = ''
    dir_img = dir_parent + 'network_input\\'
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
    weights_path = dir_parent+dir_sub + 'Weights\\'
    training_output_path = dir_parent+dir_sub + 'training output\\'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 
        
    for CV in range(0,4):
        # Exp_ID = list_Exp_ID[CV]
        # %% Load traiming images and masks from h5 files
        val_imgs = np.zeros((num_val_per, rows, cols), dtype='float32')
        val_masks = np.zeros((num_val_per, rows, cols), dtype='uint8')
        train_imgs = np.zeros((num_train_per*(nvideo-1), rows, cols), dtype='float32')
        train_masks = np.zeros((num_train_per*(nvideo-1), rows, cols), dtype='uint8')
        list_Exp_ID_train = list_Exp_ID.copy()
        Exp_ID_val = list_Exp_ID_train.pop(CV)

        print('Loading training images and masks.')
        h5_img = h5py.File(dir_img+Exp_ID_val+'.h5', 'r')
        (nframes, Lx, Ly) = h5_img['network_input'].shape
        if nframes<num_val_per:
            num_val_per=nframes
            val_imgs = np.zeros((num_val_per, rows, cols), dtype='float32')
            val_masks = np.zeros((num_val_per, rows, cols), dtype='uint8')
        val_imgs[:,:Lx,:Ly] = np.array(h5_img['network_input'][start_frame:val_every*num_val_per:val_every])
        # val_imgs = np.pad(val_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
        h5_img.close()
        h5_mask = h5py.File(dir_mask+Exp_ID_val+'.h5', 'r')
        val_masks[:,:Lx,:Ly] = np.array(h5_mask['temporal_masks'][start_frame:val_every*num_val_per:val_every])
        h5_mask.close()

        for cnt, Exp_ID in enumerate(list_Exp_ID_train):
            h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
            train_imgs[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
                = np.array(h5_img['network_input'][start_frame:train_every*num_train_per:train_every])
            h5_img.close()
            h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
            train_masks[cnt*num_train_per:(cnt+1)*num_train_per,:Lx,:Ly] \
                = np.array(h5_mask['temporal_masks'][start_frame:train_every*num_train_per:train_every])
            h5_mask.close()

        train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=(cols==rows))
        val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


        fff = get_unet() #None

        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))

        results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=[LossAndErrorPrintingCallback()])

        fff.save_weights(weights_path+'Model_CV{}.h5'.format(CV))

        f = h5py.File(training_output_path+"training_output_CV{}.h5".format(CV), "w")
        f.create_dataset("val_loss", data=results.history['val_loss'])
        f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        f.close()

