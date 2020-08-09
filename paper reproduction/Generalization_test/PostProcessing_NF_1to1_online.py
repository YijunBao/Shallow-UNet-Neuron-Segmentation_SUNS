# %%
import os
import sys
# import io
import numpy as np
# import cv2
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import time
import h5py
import multiprocessing as mp

sys.path.insert(1, '..\\neuron_post')
sys.path.insert(1, '..\\online')
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import paremter_optimization_after
from functions_other_data import data_info_neurofinder
import functions_online

    
# %% 
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    # list_neurofinder = list_neurofinder[0:4]
    for trainset_type in {'train', 'test'}: # 
        testset_type = list({'train','test'}-{trainset_type})[0]

        nvideo = len(list_neurofinder)
        thred_std = 3
        num_train_per = 1800
        BATCH_SIZE = 20
        NO_OF_EPOCHS = 200
        frames_init = 300
        merge_every = 10
        # list_vid = list(range(0,nvideo))
        # batch_size_eval = 200
        track = False
        useWT = False

        dir_video_test = 'E:\\NeuroFinder\\{} videos\\'.format(testset_type)
        dir_video_train = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        dir_parent_test = dir_video_test + 'ShallowUNet online\\complete\\'
        dir_parent_train = dir_video_train + 'ShallowUNet online\\complete\\'
        dir_sub = '1to1\\DL+BCE\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        dir_pmap = dir_parent_test + dir_sub + 'test\\probability map\\'#
        # dir_GTMasks = 'C:\\Matlab Files\\STNeuroNet-master\\Markings\\Neurofinder\\train\\Grader1\\FinalMasks_' 
        dir_GTMasks = dir_video_test + 'GTMasks_'
        dir_output = dir_parent_test + dir_sub + 'test\\output_masks batch\\' #
        dir_params = dir_parent_train + dir_sub + 'train\\params batch\\' #
        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 
        p = mp.Pool()


    # %%
        for (eid,Exp_ID) in enumerate(list_neurofinder): # list(range(0,6))+list(range(7,10)): #
            # if eid!=5:
            #     continue
            if trainset_type == 'train':
                Exp_ID_train = Exp_ID
                Exp_ID = Exp_ID + '.test'
            elif trainset_type == 'test':
                Exp_ID_train = Exp_ID + '.test'
                Exp_ID = Exp_ID
            print('Video ', Exp_ID)

            # %%    
            print('load probability map {}'.format(Exp_ID))
            start = time.time()
            f = h5py.File(dir_pmap + '{}.h5'.format(Exp_ID), "r")
            pmaps = np.array(f["probability_map"]) #[:, :Lx, :Ly]
            f.close()
            end = time.time()
            print('load time: {} s'.format(end - start))
            (nframes, Lx, Ly) = pmaps.shape
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'

            Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(Exp_ID_train))
            Params_post_mat = Optimization_Info['Params'][0]
            Params={'minArea': Params_post_mat['minArea'][0][0,0], 
                'avgArea': Params_post_mat['avgArea'][0][0,0],
                'thresh_pmap': Params_post_mat['thresh_pmap'][0][0,0], 
                'thresh_mask': Params_post_mat['thresh_mask'][0][0,0], 
                'thresh_COM0': Params_post_mat['thresh_COM0'][0][0,0], 
                'thresh_COM': Params_post_mat['thresh_COM'][0][0,0], 
                'thresh_IOU': Params_post_mat['thresh_IOU'][0][0,0], 
                'thresh_consume': Params_post_mat['thresh_consume'][0][0,0], 
                'cons':Params_post_mat['cons'][0][0,0]}
            # thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
            print(Params)

            # %% Apply the optimal parameters to the test dataset
            start = time.time()
            Masks_2 = functions_online.complete_segment_online(pmaps, Params, (Lx,Ly), \
                frames_init, merge_every, display=True, track=track, useWT=useWT, p=p)
            finish = time.time()
            Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly))
            data_GT=loadmat(filename_GT)
            GTMasks_2 = data_GT['GTMasks_2'].transpose()
            (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ=0.5)
            print({'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time': finish-start})
            time_per_frame = (finish-start)/pmaps.shape[0]*1000
            Info_dict = {'Params':Params, 'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time_frame': time_per_frame}
            savemat(dir_output+'Output_Info_{}.mat'.format(Exp_ID), Info_dict)
            savemat(dir_output+'Output_Masks_{}.mat'.format(Exp_ID), {'Masks_2':Masks_2})
        # %%
        p.close()
