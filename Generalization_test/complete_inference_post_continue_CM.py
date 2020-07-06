# %%
import sys
import os
import random
import time
import glob
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
sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')

from unet4_best import get_unet
from par2 import fastuint, fastcopy
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import paremter_optimization_after, paremter_optimization_WT_after, paremter_optimization_before

# %%
if __name__ == '__main__':
    list_name_video = ['J115', 'J123', 'K53', 'YST']
    list_radius = [8,10,8,6] # 
    list_rate_hz = [30,30,30,10] # 
    list_decay_time = [0.4, 0.5, 0.4, 0.75]
    Dimens = [(224,224),(216,152), (248,248),(120,88)]
    list_nframes = [90000, 41000, 116043, 3000]
    ID_part = ['_part11', '_part12', '_part21', '_part22']
    list_Mag = [x/8 for x in list_radius]

    thred_std = 6
    num_train_per = 2400
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    batch_size_eval = 100
    useWT=False

    for ind_video in [0, 2]: # range(0,3): # 
        name_video = list_name_video[ind_video]
        list_Exp_ID = [name_video+x for x in ID_part]
        dir_video = 'F:\\CaImAn data\\WEBSITE\\divided_data\\'+name_video+'\\'
        dir_GTMasks = dir_video + 'FinalMasks_'
        nvideo = len(list_Exp_ID)
        # nn = 23200
        # size = 496
        (rows, cols) = Dimens[ind_video] # size of the network input and output
        (Lx, Ly) = (rows, cols) # size of the original video

        dir_parent = dir_video + 'ShallowUNet\\complete\\'
        dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+20BCE\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        dir_img = dir_parent + 'network_input\\'
        dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
        weights_path = dir_parent + dir_sub + 'Weights\\'
        dir_output = dir_parent + dir_sub + 'output_masks\\' #
        dir_temp = dir_parent + dir_sub + 'temp\\'#
        Time_per_frame = np.zeros((nvideo,nvideo))
        # p = mp.Pool(mp.cpu_count())

        # %% 
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp) 

        list_minArea = list(range(60,125,10)) # [90] # must be in ascend order
        # list_minArea = list(range(80,145,10)) # [90] # must be in ascend order
        list_avgArea = [177] # list(range(130,165,10)) #
        # list_thresh_pmap = list(range(244,253,2)) #list(range(230,248,4)) # [254] # 
        list_thresh_pmap = list(range(140,235,10))
        thresh_mask = 0.5
        thresh_COM0 = 2
        list_thresh_COM = list(np.arange(4, 9, 1)) #[5]#
        list_thresh_IOU = [0.5]#list(np.arange(0.3, 0.85, 0.1)) #
        # list_thresh_consume = [(1+x)/2 for x in list_thresh_IOU]
        list_cons = list(range(1, 8, 1)) #[1] #
        list_win_avg = [1]#list(range(1, 13, 3)) #
        # consecutive = 'after'

        Mxy = list_Mag[ind_video]
        list_minArea= list(np.round(np.array(list_minArea) * Mxy**2))
        list_avgArea= list(np.round(np.array(list_avgArea) * Mxy**2))
        # list_thresh_pmap= list(np.array(list_thresh_pmap))
        # list_win_avg=list(np.array(list_win_avg))
        # thresh_mask= thresh_mask
        thresh_COM0= thresh_COM0 * Mxy
        list_thresh_COM= list(np.array(list_thresh_COM) * Mxy)
        # list_thresh_IOU= list(np.array(list_thresh_IOU)), 
        list_cons=list(np.round(np.array(list_cons) * list_rate_hz[ind_video]/30).astype('int'))

        Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
                'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
                'thresh_mask': thresh_mask, 'list_cons': list_cons, 'list_win_avg': list_win_avg}
        print(Params_set)

        size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
        F1_train = np.zeros(size_F1)
        Recall_train = np.zeros(size_F1)
        Precision_train = np.zeros(size_F1)
        (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
            =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
            # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

        for (eid,Exp_ID) in enumerate(list_Exp_ID): #[::-1][1:]
            list_saved_results = glob.glob(dir_temp+'Parameter Optimization * Exp{}.mat'.format(Exp_ID))
            p = mp.Pool(mp.cpu_count())
            if len(list_saved_results)<9:
                # eid = 9-eid
                # Exp_ID = list_Exp_ID[CV]
                test_imgs = 0
                print('Video '+Exp_ID)
                start = time.time()
                h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
                (nframes, dims0, dims1) = h5_img['network_input'].shape
                test_imgs = np.zeros((nframes, rows, cols, 1), dtype='float32')
                for t in range(nframes): # use this one to save memory
                    test_imgs[t, :dims0,:dims1,0] = np.array(h5_img['network_input'][t])
                # test_imgs[:,:dims0,:dims1,0] = np.array(h5_img['network_input']) # [:nn]
                h5_img.close()
                # test_imgs = np.pad(test_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
                nmask = 100
                h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
                test_masks = np.zeros((nmask, rows, cols, 1), dtype='float32')
                test_masks[:,:dims0,:dims1,0] = np.array(h5_mask['temporal_masks'][:nmask])
                h5_mask.close()
                time_load = time.time()
                filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
                print('Load data: {} s'.format(time_load-start))

            list_CV = list(range(nvideo))
            if eid in list_CV:
                list_CV.pop(eid)

            for CV in list_CV:
                # if CV not in [4,8]:
                #     continue
                mat_filename = dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID)
                if os.path.exists(mat_filename):
                    mdict = loadmat(mat_filename)
                    Recall_train[CV,eid] = np.array(mdict['list_Recall'])
                    Precision_train[CV,eid] = np.array(mdict['list_Precision'])
                    F1_train[CV,eid] = np.array(mdict['list_F1'])
            
                else:
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
                    # del test_imgs

                    prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
                    # # pmaps =(prob_map*256-0.5).astype(np.uint8)
                    pmaps = np.zeros(prob_map.shape, dtype='uint8')
                    fastuint(prob_map, pmaps)
                    # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
                    # fastcopy(prob_map, pmaps)
                    # pmaps = prob_map.copy()
                    del prob_map, fff
            
                    list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT, p=p) # , eng
                    # list_Recall, list_Precision, list_F1 = paremter_optimization_WT_after(pmaps, Params_set, filename_GT) # , eng
                    Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                        array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                    
                    Recall_train[CV,eid] = list_Recall
                    Precision_train[CV,eid] = list_Precision
                    F1_train[CV,eid] = list_F1
                    mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
                    savemat(mat_filename, mdict)
                    del pmaps

            p.close()
            # del test_imgs
                
                
        # %%
        for CV in range(nvideo):
            Recall_mean = Recall_train[CV].mean(axis=0)*nvideo/(nvideo-1)
            Precision_mean = Precision_train[CV].mean(axis=0)*nvideo/(nvideo-1)
            F1_mean = F1_train[CV].mean(axis=0)*nvideo/(nvideo-1)
            Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
            print('F1_max=', [x.max() for x in F1_train])
            ind = F1_mean.argmax()
            ind = np.unravel_index(ind,F1_mean.shape)

            minArea = list_minArea[ind[0]]
            avgArea = list_avgArea[ind[1]]
            thresh_pmap = list_thresh_pmap[ind[2]]
            thresh_COM = list_thresh_COM[ind[3]]
            thresh_IOU = list_thresh_IOU[ind[4]]
            thresh_consume = (1+thresh_IOU)/2
            cons = list_cons[ind[5]]
            win_avg = 1
            Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'win_avg':win_avg, 'thresh_mask': thresh_mask, 
                'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
            print(Params)
            print('F1_mean=', F1_mean[ind])
            Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
                'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
            savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)

    # %%
        print('Inference time', Time_per_frame.mean(), '+-', Time_per_frame.std())
        savemat(dir_output+'Time_per_frame.mat',{'Time_per_frame':Time_per_frame})

