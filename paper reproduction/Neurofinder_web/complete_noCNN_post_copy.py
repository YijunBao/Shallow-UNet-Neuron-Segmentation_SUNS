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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.insert(0, '..\\PreProcessing')
sys.path.insert(0, r'C:\Matlab Files\neuron_post')
# sys.path.insert(1, '../..') # the path containing "suns" folder

from par3 import fastuint_SNR
from evaluate_post import GetPerformance_Jaccard_2
# from suns.PostProcessing.complete_post import paremter_optimization
from complete_post import paremter_optimization_after

# %%
if __name__ == '__main__':
    #-------------- Start user-defined parameters --------------#
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID_full = [['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', \
                        '00.06', '00.07', '00.08', '00.09', '00.10', '00.11'], \
                        ['01.00', '01.01'], ['02.00', '02.01'], ['03.00'], ['04.00'], ['05.00']] 
                        # '04.01' is renamed as '05.00', because the imaging condition is different from '04.00'
    list_rate_hz = [7, 7.5, 8, 7.5, 6.75, 3]
    list_px_um = [1/1.15, 1/0.8, 1/1.15, 1.17, 0.8, 1.25]

    for ind_set in [0,1,2,3,4,5]: # [3]: # 
        # %% set video parameters
        list_Exp_ID = list_Exp_ID_full[ind_set]
        rate_hz = list_rate_hz[ind_set] # frame rate of the video
        Mag = 0.785*list_px_um[ind_set] # spatial magnification compared to ABO videos (0.785 um/pixel). # Mag = 0.785 / pixel_size
        # folder of the raw videos
        dir_video = 'E:\\NeuroFinder\\web\\train videos\\' + list_Exp_ID[0][:2] + '\\'
        # folder of the ".mat" files stroing the GT masks in sparse 2D matrices. 'FinalMasks_' is a prefix of the file names. 
        dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_') 

        nvideo = len(list_Exp_ID)

        dir_parent = dir_video + 'noSFSNR\\' # folder to save all the processed data
        dir_sub = 'noCNN\\'
        dir_img = dir_parent + 'network_input\\'
        dir_output = dir_parent + dir_sub + 'output_masks\\' #
        dir_temp = dir_parent + dir_sub + 'temp\\'#
        # Time_per_frame = np.zeros((nvideo))
        # p = mp.Pool(mp.cpu_count())

        # %% 
        useWT = False
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp) 

        # list_minArea = list(range(80,135,10)) # [90] # must be in ascend order
        list_minArea = list(range(80,235,20)) # [90] # must be in ascend order
        list_avgArea = [177] # list(range(130,165,10)) #
        # list_thresh_pmap = list(range(244,253,2)) # list(range(230,248,4)) #[254] # 
        list_thresh_pmap = list(np.arange(48, 165, 16))
        thresh_mask = 0.5
        thresh_COM0 = 2
        list_thresh_COM = list(np.arange(4, 9, 1)) #[5]#
        list_thresh_IOU = [0.5]#list(np.arange(0.3, 0.85, 0.1)) #
        # list_thresh_consume = [(1+x)/2 for x in list_thresh_IOU]
        list_cons = list(range(1, 8, 1)) #[1] #
        list_win_avg = [1]#list(range(1, 13, 3)) #
        # consecutive = 'after'
        # adjust the units of the hyper-parameters to pixels in the test videos according to relative magnification
        list_minArea= list(np.round(np.array(list_minArea) * Mag**2))
        list_avgArea= list(np.round(np.array(list_avgArea) * Mag**2))
        thresh_COM0= thresh_COM0 * Mag
        list_thresh_COM= list(np.array(list_thresh_COM) * Mag)
        Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
                'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
                'thresh_mask': thresh_mask, 'list_cons': list_cons, 'list_win_avg': list_win_avg}
        print(Params_set)

        size_F1 = (nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
        F1_train = np.zeros(size_F1)
        Recall_train = np.zeros(size_F1)
        Precision_train = np.zeros(size_F1)
        (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
            =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
            # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.
        list_F1 = np.zeros(nvideo)
        list_Recall = np.zeros(nvideo)
        list_Precision = np.zeros(nvideo)

        for (eid,Exp_ID) in enumerate(list_Exp_ID): #[::-1][1:]
            # list_saved_results = glob.glob(dir_temp+'Parameter Optimization Exp{}.mat'.format(Exp_ID))
            p = mp.Pool(mp.cpu_count())
            # if not len(list_saved_results):
                # eid = 9-eid
                # Exp_ID = list_Exp_ID[CV]

            # list_CV = list(range(nvideo))
            # if eid in list_CV:
            #     list_CV.pop(eid)

            # for CV in list_CV:
            # if CV not in [4,8]:
            #     continue
            mat_filename = dir_temp+'Parameter Optimization Exp{}.mat'.format(Exp_ID)
            if os.path.exists(mat_filename):
                pass
                # mdict = loadmat(mat_filename)
                # Recall_train[eid] = np.array(mdict['list_Recall'])
                # Precision_train[eid] = np.array(mdict['list_Precision'])
                # F1_train[eid] = np.array(mdict['list_F1'])
        
            else:
                test_imgs = 0
                print('Video '+Exp_ID)
                filename_GT = dir_GTMasks + Exp_ID + '.mat'
                data_GT=loadmat(filename_GT)
                GTMasks = data_GT['FinalMasks']
                (Ly, Lx, _) = GTMasks.shape

                start = time.time()
                h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
                test_imgs = np.array(h5_img['network_input'][:, :Lx, :Ly])
                h5_img.close()

                time_load = time.time()
                filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
                print('Load data: {} s'.format(time_load-start))
                start = time.time()

                # prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
                # # pmaps =(prob_map*256-0.5).astype(np.uint8)
                SNR = np.zeros(test_imgs.shape, dtype='uint8')
                # fastuint(prob_map, pmaps)
                fastuint_SNR(test_imgs, SNR, 32)
                # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
                # fastcopy(prob_map, pmaps)
                # pmaps = prob_map.copy()
                del test_imgs
        
                list_Recall, list_Precision, list_F1 = paremter_optimization_after(SNR, Params_set, filename_GT, useWT=useWT, p=p) # , eng
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                
                Recall_train[eid] = list_Recall
                Precision_train[eid] = list_Precision
                F1_train[eid] = list_F1
                mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
                savemat(mat_filename, mdict)
                del SNR

            p.close()
            # del test_imgs
                
                
        # %%
        for CV in [nvideo]: # range(nvideo):
            list_ind = np.array([x for x in range(nvideo) if x!=CV])
            Recall_mean = Recall_train[list_ind].mean(axis=0)
            Precision_mean = Precision_train[list_ind].mean(axis=0)
            F1_mean = F1_train[list_ind].mean(axis=0)
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
                'Recall_train':Recall_train, 'Precision_train':Precision_train, 'F1_train':F1_train}
                # 'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
            savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)

            # list_F1[CV] = F1_train[CV][ind]
            # list_Recall[CV] = Recall_train[CV][ind]
            # list_Precision[CV] = Precision_train[CV][ind]

        # Info_dict = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1}
        # savemat(dir_output+'Output_Info_All_self.mat', Info_dict)

    # %%
        # print('Inference time', Time_per_frame.mean(), '+-', Time_per_frame.std())
        # savemat(dir_output+'Time_per_frame.mat',{'Time_per_frame':Time_per_frame})

