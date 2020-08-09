# %%
import os
import sys
import random
import numpy as np
import cv2
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
import h5py

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(1, '..\\Network')
from data_gen_NF import data_gen_rectangle
# from train_float32 import data_gen
from unet4_modified import get_unet
from functions_other_data import data_info_neurofinder


# %%
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder = list_neurofinder[0:4]
    nvideo = len(list_neurofinder)
    thred_std = 3
    num_train_per = 1800
    num_val_per = num_train_per//10
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    list_vid = list(range(0,nvideo))

    for trainset_type in {'train', 'test'}: # 

        dir_video_train = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        dir_video_val = 'E:\\NeuroFinder\\{} videos\\'.format(list({'train','test'}-{trainset_type})[0])
        dir_parent = dir_video_train + 'ShallowUNet\\noSF\\'
        dir_parent_val = dir_video_val + 'ShallowUNet\\noSF\\'
        dir_sub = '1to1\\DL+BCE\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        dir_img = dir_parent + 'network_input\\'
        dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
        dir_img_val = dir_parent_val + 'network_input\\'
        dir_mask_val = dir_parent_val + 'temporal_masks({})\\'.format(thred_std)
        weights_path = dir_parent+dir_sub + 'train\\Weights\\'
        training_output_path = dir_parent+dir_sub + 'train\\training output\\'
        if not os.path.exists(weights_path):
            os.makedirs(weights_path) 
        if not os.path.exists(training_output_path):
            os.makedirs(training_output_path) 
            
        NO_OF_TRAINING_IMAGES = num_train_per*1
        NO_OF_VAL_IMAGES = num_val_per*1
        for vid in list_vid:
            # if vid!=5:
            #     continue
            Exp_ID = list_neurofinder[vid]
            if trainset_type == 'train':
                Exp_ID_train = Exp_ID
                Exp_ID_val = Exp_ID + '.test'
            elif trainset_type == 'test':
                Exp_ID_train = Exp_ID + '.test'
                Exp_ID_val = Exp_ID
            # fname_info = dir_video_train + 'neurofinder.' + Exp_ID + '\\info.json'
            # info, Mxy = data_info_neurofinder(fname_info)
            # (nframes, rows, cols) = dims_raw = info['dimensions'][::-1]

            # %% Load traiming images and masks from h5 files
            print('Loading training images and masks.')
            h5_img = h5py.File(dir_img+Exp_ID_train+'.h5', 'r')
            (nframes, rows, cols) = h5_img['network_input'].shape
            train_every = nframes//num_train_per
            start_frame = random.randint(0,train_every-1)
            train_imgs = np.array(h5_img['network_input'][start_frame:train_every*num_train_per:train_every])
            h5_img.close()
            h5_mask = h5py.File(dir_mask+Exp_ID_train+'.h5', 'r')
            train_masks = np.array(h5_mask['temporal_masks'][start_frame:train_every*num_train_per:train_every])
            h5_mask.close()

            h5_img = h5py.File(dir_img_val + Exp_ID_val + '.h5', 'r')
            (nframes, rows, cols) = h5_img['network_input'].shape
            val_every = nframes//num_val_per
            start_frame = random.randint(0,val_every-1)
            val_imgs = np.array(h5_img['network_input'][start_frame:val_every*NO_OF_VAL_IMAGES:val_every])
            # val_imgs = np.pad(val_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
            h5_img.close()
            h5_mask = h5py.File(dir_mask_val + Exp_ID_val + '.h5', 'r')
            val_masks = np.array(h5_mask['temporal_masks'][start_frame:val_every*NO_OF_VAL_IMAGES:val_every])
            h5_mask.close()
            # train_imgs = np.zeros((NO_OF_TRAINING_IMAGES, rows, cols, 1), dtype='float32')
            # train_masks = np.zeros((NO_OF_TRAINING_IMAGES, rows, cols, 1), dtype='uint8')

            # [train_img1, train_mask1] = dataloader(train_frame_path, train_mask_path)
            train_gen = data_gen_rectangle(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
            #train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
            val_gen = data_gen_rectangle(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)


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
            fff.save_weights(weights_path+'Model_{}.h5'.format(Exp_ID_train))

            f = h5py.File(training_output_path+"training_output_{}.h5".format(Exp_ID_train), "w")
            f.create_dataset("val_loss", data=results.history['val_loss'])
            f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
            f.create_dataset("loss", data=results.history['loss'])
            f.create_dataset("dice_loss", data=results.history['dice_loss'])
            f.close()

    # %%
