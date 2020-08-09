# %%
import sys
import os
import random
import time
import numpy as np
# import cv2
# import tensorflow as tf
import h5py
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.insert(0, '..\\PreProcessing')
sys.path.insert(0, '..\\Network')
sys.path.insert(0, '..\\neuron_post')

from functions_other_data import data_info_neurofinder
from unet4_modified import get_unet
from par2 import fastuint, fastcopy

# %%
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder = list_neurofinder[0:4] # [4:] # 
    for trainset_type in {'train', 'test'}: #
        list_dataset_type = {'train','test'} #

        nvideo = len(list_neurofinder)
        thred_std = 3
        num_train_per = 1800
        BATCH_SIZE = 20
        NO_OF_EPOCHS = 200
        list_vid = list(range(0,nvideo))
        batch_size_eval = 10

        dir_video_train = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        dir_parent = dir_video_train + 'ShallowUNet\\noSF\\'
        dir_sub = '1to1\\DL+BCE\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        weights_path = dir_parent+dir_sub + 'train\\Weights\\'
        # p = mp.Pool(mp.cpu_count())

        for dataset_type in list_dataset_type:
            dir_video = 'E:\\NeuroFinder\\{} videos\\'.format(dataset_type)
            dir_parent_video = dir_video + 'ShallowUNet\\noSF\\'
            dir_img = dir_parent_video + 'network_input\\'
            dir_mask = dir_parent_video + 'temporal_masks({})\\'.format(thred_std)
            if dataset_type==trainset_type:
                dir_pmap = dir_parent_video + dir_sub + 'train\\probability map\\'#
            else:
                dir_pmap = dir_parent_video + dir_sub + 'test\\probability map\\'#
            if not os.path.exists(dir_pmap):
                os.makedirs(dir_pmap) 

            for (eid,Exp_ID) in enumerate(list_neurofinder): #[::-1][1:]
                if trainset_type == 'test':
                    Exp_ID_train = Exp_ID + '.test'
                else:
                    Exp_ID_train = Exp_ID
                if dataset_type == 'test':
                    Exp_ID = Exp_ID + '.test'
                # if eid!=5:
                #     continue
                print('Video '+Exp_ID)

                start = time.time()
                h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
                test_imgs = np.expand_dims(np.array(h5_img['network_input']), axis=-1) #[:nn]
                h5_img.close()
                # test_imgs = np.pad(test_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
                nmask = 100
                h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
                test_masks = np.expand_dims(np.array(h5_mask['temporal_masks'][:nmask]), axis=-1)
                h5_mask.close()
                time_load = time.time()
                # filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID + '_sparse.mat'
                print('Load data: {} s'.format(time_load-start))

                start = time.time()
                fff = get_unet() #size
                fff.load_weights(weights_path+'Model_{}.h5'.format(Exp_ID_train))

                fff.evaluate(test_imgs[:nmask],test_masks,batch_size=batch_size_eval)
                time_init = time.time()
                print('Initialization: {} s'.format(time_init-start))

                start_test = time.time()
                prob_map = fff.predict(test_imgs, batch_size=batch_size_eval)
                finish_test = time.time()
                Time_frame = (finish_test-start_test)/test_imgs.shape[0]*1000
                print('Average infrence time {} ms/frame'.format(Time_frame))
                # del test_imgs

                prob_map = prob_map.squeeze(axis=-1)#[:,:Lx,:Ly]
                # # pmaps =(prob_map*256-0.5).astype(np.uint8)
                pmaps = np.zeros(prob_map.shape, dtype='uint8')
                fastuint(prob_map, pmaps)
                # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
                # fastcopy(prob_map, pmaps)
                # pmaps = prob_map.copy()
                f = h5py.File(dir_pmap+Exp_ID+".h5", "w")
                f.create_dataset("probability_map", data = pmaps)
                f.close()
                time_save = time.time()
                print('Saving probability map: {} s'.format(time_save-finish_test))
                del prob_map, pmaps #, fff
            
        # p.close()
                
