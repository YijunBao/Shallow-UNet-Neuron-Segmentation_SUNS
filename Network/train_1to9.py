# %%
import os
import random
import numpy as np
import cv2
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
import h5py

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from data_gen import data_gen
from unet4_best_FL import get_unet


# %%
if __name__ == '__main__':
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    nvideo = len(list_Exp_ID)
    size = 488
    rows = cols = size
    thred_std = 6
    num_total = 20000
    num_train_per = 1800
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    train_every = num_total//num_train_per
    start_frame = random.randint(0,train_every-1)
    # Train the model
    num_val_per = num_train_per//90
    val_every = num_total//num_val_per
    start_frame_val = random.randint(0,val_every-1)
    NO_OF_TRAINING_IMAGES = num_train_per
    NO_OF_VAL_IMAGES = num_val_per*(nvideo-1)

    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\noSF\\'
    dir_sub = '1to9\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_img = dir_parent + 'network_input\\'
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
    weights_path = dir_parent+dir_sub + 'Weights\\'
    training_output_path = dir_parent+dir_sub + 'training output\\'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 
        
    for CV in range(10):
        # Exp_ID = list_Exp_ID[CV]
        # %% Load traiming images and masks from h5 files
        # val_imgs = np.zeros((num_train_per, rows, cols, 1), dtype='uint8')
        # val_masks = np.zeros((num_train_per, rows, cols, 1), dtype='uint8')
        val_imgs = np.zeros((NO_OF_VAL_IMAGES, rows, cols), dtype='float32')
        val_masks = np.zeros((NO_OF_VAL_IMAGES, rows, cols), dtype='uint8')
        list_Exp_ID_val = list_Exp_ID.copy()
        Exp_ID_train = list_Exp_ID_val.pop(CV)

        print('Loading training images and masks.')
        h5_img = h5py.File(dir_img+Exp_ID_train+'.h5', 'r')
        train_imgs = np.array(h5_img['network_input'][start_frame:train_every*num_train_per:train_every])
        # val_imgs = np.pad(val_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
        h5_img.close()
        h5_mask = h5py.File(dir_mask+Exp_ID_train+'.h5', 'r')
        train_masks = np.array(h5_mask['temporal_masks'][start_frame:train_every*num_train_per:train_every])
        h5_mask.close()

        for cnt, Exp_ID in enumerate(list_Exp_ID_val):
            h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
            val_imgs[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
                = np.array(h5_img['network_input'][start_frame_val:val_every*num_val_per:val_every])
            h5_img.close()
            h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
            val_masks[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
                = np.array(h5_mask['temporal_masks'][start_frame_val:val_every*num_val_per:val_every])
            h5_mask.close()

        # [train_img1, train_mask1] = dataloader(train_frame_path, train_mask_path)
        train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
        #train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
        val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


        fff = get_unet() #None

        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))

        results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=[LossAndErrorPrintingCallback()])

        """results = fff.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                                steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                validation_data=val_gen,
                                validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                                callbacks=callbacks_list)"""
        fff.save_weights(weights_path+'Model_CV{}.h5'.format(CV))

        f = h5py.File(training_output_path+"training_output_CV{}.h5".format(CV), "w")
        f.create_dataset("val_loss", data=results.history['val_loss'])
        f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        f.close()

# %%
