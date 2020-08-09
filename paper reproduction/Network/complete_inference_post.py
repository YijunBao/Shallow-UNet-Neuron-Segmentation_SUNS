# %%
import sys
import os
import random
import time
import numpy as np

import h5py
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import multiprocessing as mp

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.insert(0, '..\\PreProcessing')
# sys.path.insert(0, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
os.environ['KERAS_BACKEND'] = 'tensorflow'

from unet4_best_FL import get_unet
from par2 import fastuint, fastcopy
# from evaluate_post import GetPerformance_Jaccard_2
from complete_post import paremter_optimization_after

# %%
if __name__ == '__main__':
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    nvideo = len(list_Exp_ID)
    nn = 23200
    size = 488
    rows = cols = size
    thred_std = 3
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    batch_size_eval = 200
    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\noSF\\'
    dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+100FL(1,0.25)\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_img = dir_parent + 'network_input\\'
    dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
    weights_path = dir_parent + dir_sub + 'Weights\\'
    dir_output = dir_parent + dir_sub + 'output_masks\\' #
    dir_temp = dir_parent + dir_sub + 'temp\\'#
    Time_per_frame = np.zeros((nvideo,nvideo))
    # p = mp.Pool(mp.cpu_count())

    # %% 
    Lx = Ly = 487
    useWT = False
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    # list_minArea = list(range(80,135,10)) # [90] # must be in ascend order
    list_minArea = list(range(80,135,10)) # [90] # must be in ascend order
    list_avgArea = [177] # list(range(130,165,10)) #
    # list_thresh_pmap = list(range(244,253,2)) # list(range(230,248,4)) #[254] # 
    list_thresh_pmap = list(range(165,210,5))
    thresh_mask = 0.5
    thresh_COM0 = 2
    list_thresh_COM = list(np.arange(4, 9, 1)) #[5]#
    list_thresh_IOU = [0.5]#list(np.arange(0.3, 0.85, 0.1)) #
    # list_thresh_consume = [(1+x)/2 for x in list_thresh_IOU]
    list_cons = list(range(1, 8, 1)) #[1] #
    # consecutive = 'after'
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
             'thresh_mask': thresh_mask, 'list_cons': list_cons}
    print(Params_set)

    size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
    F1_train = np.zeros(size_F1)
    Recall_train = np.zeros(size_F1)
    Precision_train = np.zeros(size_F1)
    (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
        =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
        # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

    for (eid,Exp_ID) in enumerate(list_Exp_ID): #[::-1][1:]
        if eid>4:
            continue
        p = mp.Pool(mp.cpu_count())
        # eid = 9-eid
        # Exp_ID = list_Exp_ID[CV]
        test_imgs = 0
        print('Video '+Exp_ID)
        start = time.time()
        h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
        test_imgs = np.expand_dims(np.array(h5_img['network_input'][:nn]), axis=-1)
        h5_img.close()
        # test_imgs = np.pad(test_imgs, ((0,0),(0,1),(0,1),(0,0)),'constant', constant_values=(0, 0))
        nmask = 100
        h5_mask = h5py.File(dir_mask+Exp_ID+'.h5', 'r')
        test_masks = np.expand_dims(np.array(h5_mask['temporal_masks'][:nmask]), axis=-1)
        h5_mask.close()
        time_load = time.time()
        filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID + '_sparse.mat'
        print('Load data: {} s'.format(time_load-start))

        list_CV = list(range(nvideo))
        if eid in list_CV:
            list_CV.pop(eid)

        for CV in list_CV:
            # if CV not in [4,8]:
            #     continue
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
            Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
            
            Recall_train[CV,eid] = list_Recall
            Precision_train[CV,eid] = list_Precision
            F1_train[CV,eid] = list_F1
            mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
            savemat(dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID), mdict)
            del pmaps

            # mdict = loadmat(dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID))
            # Recall_train[CV,eid] = np.array(mdict['list_Recall'])
            # Precision_train[CV,eid] = np.array(mdict['list_Precision'])
            # F1_train[CV,eid] = np.array(mdict['list_F1'])
        
        p.close()
        # del test_imgs
            
            
    # %%
    for CV in range(nvideo):
        Recall_mean = Recall_train[CV].mean(axis=0)*10/9
        Precision_mean = Precision_train[CV].mean(axis=0)*10/9
        F1_mean = F1_train[CV].mean(axis=0)*10/9
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
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        print('F1_mean=', F1_mean[ind])
        Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
            'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
        savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)

# %%
    print('Inference time', Time_per_frame.mean(), '+-', Time_per_frame.std())
    savemat(dir_output+'Time_per_frame.mat',{'Time_per_frame':Time_per_frame})

