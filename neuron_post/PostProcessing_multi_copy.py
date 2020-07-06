# %%
import os
# import io
import numpy as np
# import cv2
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import time
import h5py
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

# from seperate import *
# from combine import *
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import paremter_optimization_after, paremter_optimization_WT_after, \
    paremter_optimization_before, complete_segment_pre

    
# %% 
if __name__ == '__main__':
    # time.sleep(3600)
    # %% 
    Lx = 487
    Ly = 487
    useWT = False
    thred_std = 6
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\noSFSNR\\'
    dir_sub = 'DL\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    dir_pmap = dir_parent + dir_sub + 'probability_map\\'
    dir_output = dir_parent + dir_sub + 'output_masks\\' #
    dir_temp = dir_parent + dir_sub + 'temp\\'#
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    # %%   
    list_minArea = list(range(80,195,10)) # [120] # must be in ascend order
    list_avgArea = [177] # list(range(130,165,10)) #
    list_thresh_pmap = list(range(10,100,10)) # + list(range(246,255,2)) # [234] # 
    thresh_mask = 0.5
    thresh_COM0 = 2
    list_thresh_COM = list(np.arange(4, 9, 1)) # [4] # 
    list_thresh_IOU = [0.5] # list(np.arange(0.3, 0.85, 0.1)) #
    # list_thresh_consume = [(1+x)/2 for x in thresh_IOU]
    list_cons = list(range(1, 8, 1)) # [4] # 
    list_win_avg = [1]#list(range(1, 13, 3)) #
    # list_cons = [1] #list(range(1, 13, 3)) #
    # list_win_avg = [2] + list(range(1, 8, 2)) #[1]#
    if len(list_win_avg)==1:
        consecutive = 'after' 
    elif len(list_cons)==1:
        consecutive = 'before'
    else:
        raise NotImplementedError('Either win_avg or cons should be one.')
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
             'thresh_mask': thresh_mask, 'list_cons': list_cons, 'list_win_avg': list_win_avg}
    print(Params_set)
    # CV = 9

# %%
    for CV in list(range(0,10)): # [6]: # +list(range(7,10))
        list_Exp_ID_train = list_Exp_ID.copy()
        Exp_ID_test = list_Exp_ID_train.pop(CV)    
        F1_train = []
        Recall_train, Precision_train = [], []
        for Exp_ID in list_Exp_ID_train: #[4:]
            # saved = loadmat(dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID))
            # F1_train.append(saved['list_F1'])
            # Recall_train.append(saved['list_Recall'])
            # Precision_train.append(saved['list_Precision'])

            # Exp_ID = list_Exp_ID[CV]
            print('load video {}'.format(Exp_ID))
            start = time.time()
            f = h5py.File(dir_pmap+'CV{}\\'.format(CV)+Exp_ID+".h5", "r")
            pmaps = np.array(f["probability_map"][:, :Lx, :Ly])
            f.close()
            end = time.time()
            print('load time: {} s'.format(end - start))
            filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID + '_sparse.mat'
            # eng = engine.connect_matlab('MATLABEngine')

            if consecutive == 'before': # paremter_optimization_before
                list_Recall, list_Precision, list_F1 = paremter_optimization_before(pmaps, Params_set, filename_GT, useWT=useWT) #, eng
                (array_AvgArea, array_minArea, array_thresh_pmap, array_win_avg, array_thresh_COM, array_thresh_IOU)\
                    =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_win_avg)
                    # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_win_avg.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T

            elif consecutive == 'after': # paremter_optimization_after
                list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT) # , eng
                # list_Recall, list_Precision, list_F1 = paremter_optimization_WT_after(pmaps, Params_set, filename_GT) # , eng
                (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
                    =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
                    # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
            
            Recall_train.append(list_Recall)
            Precision_train.append(list_Precision)
            F1_train.append(list_F1)
            mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
            savemat(dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID), mdict)
            # eng.exit()
            del pmaps


# %%
        Recall_train = np.array(Recall_train)
        Precision_train = np.array(Precision_train)
        F1_train = np.array(F1_train)
        Recall_mean = Recall_train.mean(axis=0)
        Precision_mean = Precision_train.mean(axis=0)
        F1_mean = F1_train.mean(axis=0)
        if consecutive == 'before':
            Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_win_avg.ravel(), 
                array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
        elif consecutive == 'after':
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
        if consecutive == 'before':
            win_avg = list_win_avg[ind[5]]
            cons = 1
        elif consecutive == 'after': 
            win_avg = 1
            cons = list_cons[ind[5]]
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'win_avg':win_avg, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        print('F1_mean=', F1_mean[ind])
        Info_dict = {'Params_set':Params_set, 'Params':Params, 'Recall_train':Recall_train, 'Precision_train':Precision_train, 'F1_train':F1_train, 'Table': Table}
        savemat(dir_output+'Optimization_Info_{}.mat'.format(Exp_ID_test), Info_dict)

        # %% Apply the optimal parameters to the test dataset
        f = h5py.File(dir_pmap+'CV{}\\'.format(CV)+Exp_ID_test+".h5", "r")
        pmaps = np.array(f["probability_map"][:, :Lx, :Ly])
        f.close()
        p = mp.Pool()
        start = time.time()
        Masks_2 = complete_segment_pre(pmaps, Params, display=True, useWT=useWT, p=p)
        # Masks = np.reshape(Masks_2.todense().A, (Masks_2.shape[0], Lx, Ly))
        finish = time.time()
        p.close()
        filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID_test + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time': finish-start})
        time_per_frame = (finish-start)/pmaps.shape[0]*1000
        Info_dict = {'Params':Params, 'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time_frame': time_per_frame}
        savemat(dir_output+'Output_Info_{}.mat'.format(Exp_ID_test), Info_dict)
        savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID_test), {'Masks_2':Masks_2})
