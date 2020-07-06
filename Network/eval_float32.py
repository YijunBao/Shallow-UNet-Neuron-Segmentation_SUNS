# %%
import os
import random
import time
import numpy as np
import cv2
import tensorflow as tf
from unet4_best import get_unet
import os
import h5py
from par2 import fastuint

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%
if __name__ == '__main__':
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    nvideo = len(list_Exp_ID)
    thred_std = 7
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\complete\\'
    dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+20BCE\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_img = dir_parent + 'network_input\\'
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
    weights_path = dir_parent + dir_sub + 'Weights\\'
    dir_pmap = dir_parent + dir_sub + 'probability_map\\'
    if not os.path.exists(dir_pmap):
        os.makedirs(dir_pmap) 
        
    batch_size_eval = 200
    size = 488
    rows = cols = size
    Time_per_frame = np.zeros((10,10))

    for (eid,Exp_ID) in enumerate(list_Exp_ID): #[:2]
        if eid<=4:
            continue
        # Exp_ID = list_Exp_ID[CV]
        print('Video '+Exp_ID)
        start = time.time()
        h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        test_imgs = np.expand_dims(np.array(h5_img['network_input']), axis=-1)
        h5_img.close()
        # test_imgs = np.pad(test_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
        nmask = 100
        h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        test_masks = np.expand_dims(np.array(h5_mask['temporal_masks'][:nmask]), axis=-1)
        h5_mask.close()
        time_load = time.time()
        print('Load data: {} s'.format(time_load-start))

        for CV in [eid]: # range(0,10):
            dir_pmap_CV = dir_pmap+'CV{}\\'.format(CV)
            if not os.path.exists(dir_pmap_CV):
                os.makedirs(dir_pmap_CV)
            start = time.time()
            fff = get_unet() #size
            fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))

            fff.evaluate(test_imgs[:nmask],test_masks,batch_size=batch_size_eval)
            time_init = time.time()
            print('Initialization: {} s'.format(time_init-start))

            start_test = time.time()
            prob_map = fff.predict(test_imgs, batch_size=batch_size_eval)
            finish_test = time.time()
            Time_frame = (finish_test-start_test)/test_imgs.shape[0]*1000
            print('Average infrence time {} ms/frame'.format(Time_frame))
            Time_per_frame[CV,eid] = Time_frame

            prob_map = prob_map.squeeze(axis=-1)
            # pmap =(prob_map*256-0.5).astype(np.uint8)
            pmap = np.zeros(prob_map.shape, dtype='uint8')
            fastuint(prob_map, pmap)
            f = h5py.File(dir_pmap_CV+Exp_ID+".h5", "w")
            f.create_dataset("probability_map", data = pmap)
            f.close()
            time_save = time.time()
            print('Saving probability map: {} s'.format(time_save-finish_test))
            del prob_map, pmap
    
# %%
    print(Time_per_frame.mean(), Time_per_frame.std())
    np.save(dir_pmap+'Time_per_frame.npy',Time_per_frame)