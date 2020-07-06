# %%
import os
import sys
import numpy as np
# import cv2
from scipy import sparse
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import time
import h5py
import multiprocessing as mp
# import matlab
# import matlab.engine as engine

sys.path.insert(1, '..\\neuron_post')
# from seperate import *
# from combine import *
from evaluate_post import GetPerformance_Jaccard_2
# from complete_post import paremter_optimization_after, paremter_optimization_before, complete_segment
from seperate_multi import separateNeuron_noWT, watershed_neurons, separateNeuron, separateNeuron_b
from par3 import fastthreshold
import functions_online

    
# %% 
if __name__ == '__main__':
    Lx = 487
    Ly = 487
    ## %% 
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    # dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\'
    # dir_pmap = dir_parent+'probability_map\\'
    useWT = False
    track = False
    # show_intermediate = False
    # nvideo = len(list_Exp_ID)
    thred_std = 7
    num_train_per = 200
    BATCH_SIZE = 20
    NO_OF_EPOCHS = 200
    frames_init = 900
    merge_every = 30
    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\complete\\'
    dir_sub = 'std{}_nf{}_ne{}_bs{}\\DL+20BCE\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
    # dir_img = dir_parent + 'network_input\\'
    # dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std)
    # weights_path = dir_parent + dir_sub + 'Weights\\'
    dir_pmap = dir_parent + dir_sub + 'probability_map\\'

    ## %%
    for CV in [8]: # range(0,10):
        Exp_ID_test = list_Exp_ID[CV]
        minArea = 120
        avgArea = 177
        thresh_pmap = 234
        thresh_COM = 6
        thresh_IOU = 0.5
        thresh_consume = (1+thresh_IOU)/2
        win_avg = 1
        cons = 4
        thresh_COM0 = 2
        thresh_mask = 0.5
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'win_avg':win_avg, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        thresh_pmap_float = (thresh_pmap+1.5)/256

        # %% Apply the optimal parameters to the test dataset
        f = h5py.File(dir_pmap+'CV{}\\'.format(CV)+Exp_ID_test+".h5", "r")
        pmaps = np.array(f["probability_map"][:, :Lx, :Ly])
        # pmaps = np.array(f["probability_map"][:4000, 400:487, 200:300])
        # pmaps = np.array(f["probability_map"][:4000, 200:Lx, 200:Ly])
        (nframes, Lx, Ly) = pmaps.shape
        f.close()
        p = mp.Pool()
        start = time.time()
        # pmaps_b = np.zeros(pmaps.shape, dtype='uint8')
        # fastthreshold(pmaps, pmaps_b, thresh_pmap_float)

        Masks_2 = functions_online.complete_segment_online(pmaps, Params, (Lx,Ly), \
            frames_init, merge_every, display=True, track=track, useWT=useWT, p=p)
        # segs = p.starmap(separateNeuron, [(frame, thresh_pmap, minArea, avgArea, useWT) for frame in pmaps], chunksize=1) #, eng
        # # Masks_2, _, _, _ = functions_online.merge_complete(segs, (Lx,Ly), Params)

        # tuple_init = functions_online.merge_complete(segs[:frames_init], (Lx,Ly), Params)

        # # init = time.time()
        # # %% 
        # if track:
        #     Masks_2 = functions_online.merge_final_track(tuple_init, segs[frames_init:], (Lx,Ly), Params, frames_init, merge_every) #, show_intermediate=True
        # else:
        #     Masks_2 = functions_online.merge_final(tuple_init, segs[frames_init:], (Lx,Ly), Params, frames_init, merge_every) #, show_intermediate=True
        Masks = np.reshape(Masks_2.todense().A, (Masks_2.shape[0], Lx, Ly))
        finish = time.time()
        # print(Masks_2_track.shape, Masks_2.shape)
        # print(len(tuple_temp_track[0]), len(tuple_temp[0]))

        # %% 
        p.close()
        print('Post time:', finish-start)
        filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID_test + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time': finish-start})

# %% used for debug
#         masks_show_track = sum(tuple_temp_track[0]).toarray().reshape(Lx,Ly)
#         plt.imshow(masks_show_track>0); plt.show()
#         masks_show = sum(tuple_temp[0]).toarray().reshape(Lx,Ly)
#         plt.imshow(masks_show>0); plt.show()
# # %%
#         masks_nocons_track = [el for (iscons,el) in zip(tuple_temp_track[4],tuple_temp_track[0]) if not iscons]
#         masks_show_track = sum(masks_nocons_track).toarray().reshape(Lx,Ly)
#         plt.imshow(masks_show_track>0); plt.show()
#         masks_nocons = [el for (iscons,el) in zip(tuple_temp[4],tuple_temp[0]) if not iscons]
#         masks_show = sum(masks_nocons).toarray().reshape(Lx,Ly)
#         plt.imshow(masks_show>0); plt.show()


# # %%
#         masks_show_init = sum(tuple_init[0]).toarray().reshape(Lx,Ly)
#         plt.imshow(masks_show_init>0); plt.show()
