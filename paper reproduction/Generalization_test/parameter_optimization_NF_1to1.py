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
sys.path.insert(1, '..\\neuron_post')

from functions_other_data import data_info_neurofinder
from complete_post import paremter_optimization_after


# %%
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder = list_neurofinder[0:4]
    for trainset_type in {'train', 'test'}: # 

        nvideo = len(list_neurofinder)
        thred_std = 3
        num_train_per = 1800
        BATCH_SIZE = 20
        NO_OF_EPOCHS = 200
        # list_vid = list(range(0,6))
        # batch_size_eval = 200
        useWT = False

        dir_video = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        dir_parent = dir_video + 'ShallowUNet\\noSF\\'
        dir_sub = '1to1\\DL+BCE\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        dir_pmap = dir_parent + dir_sub + 'train\\probability map\\'#
        # dir_GTMasks = 'C:\\Matlab Files\\STNeuroNet-master\\Markings\\Neurofinder\\train\\Grader1\\FinalMasks_' 
        dir_GTMasks = dir_video + 'GTMasks_'
        dir_params = dir_parent + dir_sub + 'train\\params\\' #
        dir_temp = dir_parent + dir_sub + 'train\\temp\\'#
        if not os.path.exists(dir_params):
            os.makedirs(dir_params) 
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp) 

        # %% 
        # Lx = Ly = 487
        list_minArea = list(range(60,175,10)) # [90] # must be in ascend order
        list_avgArea = [177] # list(range(110,205,30)) #
        list_thresh_pmap = list(range(130,255,10)) # [254] # list(range(238,246,4)) + # list(np.arange(0.95,0.995, 0.01)) # 
        thresh_mask = 0.5
        thresh_COM0 = 2
        list_thresh_COM = list(np.arange(4, 11, 1)) #[5]#
        list_thresh_IOU = [0.5]#list(np.arange(0.3, 0.85, 0.1)) #
        # list_thresh_consume = [(1+x)/2 for x in list_thresh_IOU]
        list_cons = list(range(1, 8, 1)) #[1] #
        Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
                'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
                'thresh_mask': thresh_mask, 'list_cons': list_cons}
        print('Original Params_set before resizing: ', Params_set)

        size_F1 = (nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
        F1_train = np.zeros(size_F1)
        Recall_train = np.zeros(size_F1)
        Precision_train = np.zeros(size_F1)
        (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
            =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
            # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

        p = mp.Pool(mp.cpu_count())
        for (eid,Exp_ID) in enumerate(list_neurofinder): #[::-1][1:]
            if trainset_type == 'test':
                Exp_ID = Exp_ID + '.test'
            # if eid!=5:
            #     continue
            print('Video '+Exp_ID)
            fname_info = dir_video + 'neurofinder.' + Exp_ID + '\\info.json'
            info, Mxy = data_info_neurofinder(fname_info)
            # dims_raw = info['dimensions'][::-1]

            start = time.time()
            h5_pmap = h5py.File(dir_pmap+Exp_ID+'.h5', 'r')
            pmaps = np.array(h5_pmap['probability_map']) #[:nn]
            h5_pmap.close()
            time_load = time.time()
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            print('Load data: {} s'.format(time_load-start))

            Params_set={'list_minArea': list(np.round(np.array(list_minArea) * Mxy**2)), 
                'list_avgArea': list(np.round(np.array(list_avgArea) * Mxy**2)),
                'list_thresh_pmap': list(np.array(list_thresh_pmap)), 
                'thresh_mask': thresh_mask, 
                'thresh_COM0': thresh_COM0 * Mxy, 
                'list_thresh_COM': list(np.array(list_thresh_COM) * Mxy), 
                'list_thresh_IOU': list(np.array(list_thresh_IOU)), 
                'list_cons':list(np.round(np.array(list_cons) * info['rate-hz']/30*5).astype('int'))}
                
            list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT, p=p) # , eng
            Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
            
            Recall_train[eid] = list_Recall
            Precision_train[eid] = list_Precision
            F1_train[eid] = list_F1
            mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
            savemat(dir_temp+'Parameter Optimization {}.mat'.format(Exp_ID), mdict)
            
            ind = list_F1.argmax()
            ind = np.unravel_index(ind,list_F1.shape)
            F1 = list_F1[ind]
            Precision = list_Precision[ind]
            Recall = list_Recall[ind]
            minArea = Params_set['list_minArea'][ind[0]]
            avgArea = Params_set['list_avgArea'][ind[1]]
            thresh_pmap = Params_set['list_thresh_pmap'][ind[2]]
            thresh_COM = Params_set['list_thresh_COM'][ind[3]]
            thresh_IOU = Params_set['list_thresh_IOU'][ind[4]]
            thresh_consume = (1+thresh_IOU)/2
            cons = Params_set['list_cons'][ind[5]]
            Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
                'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
            print(Params)
            Info_dict={'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Params_set':Params_set, 'Params':Params, 'Table':Table}
            savemat(dir_params+'Optimization_Info_{}.mat'.format(Exp_ID), Info_dict)
            del pmaps

        p.close()