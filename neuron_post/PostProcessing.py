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
from complete_post import paremter_optimization_after, paremter_optimization_before, complete_segment

    
# %% 
if __name__ == '__main__':
    # %% 
    Lx = 487
    Ly = 487
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    dir_parent = 'D:\\ABO\\20 percent\\ShallowUNet\\'
    dir_pmap = dir_parent+'probability_map\\'

    # %%
    for CV in range(6,7):
        Exp_ID_test = list_Exp_ID[CV]
        minArea = 120
        avgArea = 130
        thresh_pmap = 234
        thresh_COM = 4
        thresh_IOU = 0.5
        thresh_consume = (1+thresh_IOU)/2
        win_avg = 1
        cons = 4
        thresh_COM0 = 2
        thresh_mask = 0.5
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'win_avg':win_avg, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)

        # %% Apply the optimal parameters to the test dataset
        f = h5py.File(dir_pmap+'CV{}\\'.format(CV)+Exp_ID_test+".h5", "r")
        pmaps = np.array(f["probability_map"][:, :Lx, :Ly])
        f.close()
        p = mp.Pool()
        start = time.time()
        Masks_2 = complete_segment(pmaps, Params, display=True, p=p)
        Masks = np.reshape(Masks_2.todense().A, (Masks_2.shape[0], Lx, Ly))
        finish = time.time()
        p.close()
        filename_GT = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID_test + '_sparse.mat'
        data_GT=loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ=0.5)
        print({'Recall':Recall, 'Precision':Precision, 'F1':F1, 'Time': finish-start})
