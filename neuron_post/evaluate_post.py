import io
import numpy as np
from scipy import sparse
from scipy import signal
from scipy.optimize import linear_sum_assignment
import time
import multiprocessing as mp
# import matlab
# import matlab.engine as engine


## Calculate F1 using MATLAB
def evaluate_segmentation_MATLAB(filename_output: str, filename_GT: str, list_cons: list, thresh_mask=0.5, doesplot=False): # eng, 
    list_mmin = [0]
    eng = engine.connect_matlab('Test')
    out = io.StringIO()
    err = io.StringIO()
    evaluate = eng.evaluate_seperate_py(filename_output, filename_GT, matlab.double(list_cons), matlab.double(list_mmin), 
                            matlab.double([thresh_mask]), matlab.logical([doesplot]), nargout=3, stdout=out, stderr=err) #
    if out.getvalue():
        print(out.getvalue())
    if err.getvalue():
        print(err.getvalue())
    Recall_k = np.array(evaluate[0])
    Precision_k = np.array(evaluate[1])
    F1_k = np.array(evaluate[2])
    ndim = F1_k.ndim
    if ndim == 0:
        Recall_k = [Recall_k]
        Precision_k = [Precision_k]
        F1_k =[F1_k]
    # elif ndim == 1: # should be unecessary, but just in case of future function change
    #     if len(list_cons)==1:
    #         Recall_k = np.expand_dims(Recall_k, 0)
    #         Precision_k = np.expand_dims(Precision_k, 0)
    #         F1_k = np.expand_dims(F1_k, 0)
    #     if len(list_mmin)==1:
    #         Recall_k = np.expand_dims(Recall_k, 1)
    #         Precision_k = np.expand_dims(Precision_k, 1)
    #         F1_k = np.expand_dims(F1_k, 1)
    eng.exit()
    return Recall_k, Precision_k, F1_k

##
def GetPerformance_Jaccard_2(GTMasks, Masks, ThreshJ=0.5):
    if 'bool' in str(Masks.dtype):
        Masks = Masks.astype('uint32')
    if 'bool' in str(GTMasks.dtype):
        GTMasks = GTMasks.astype('uint32')
    NGT = GTMasks.shape[0]
    NMask = Masks.shape[0]
    a1 = np.repeat(GTMasks.sum(axis=1).A, NMask, axis=1)
    a2 = np.repeat(Masks.sum(axis=1).A.T, NGT, axis=0)
    intersectMat = GTMasks.dot(Masks.transpose()).A
    unionMat = a1 + a2 - intersectMat
    JaccardInd = intersectMat/unionMat
    Dmat = 1-JaccardInd
    # Dmat[intersectMat == a1] = 0
    # Dmat[intersectMat == a2] = 0
    D = Dmat
    D[D > 1 - ThreshJ] = 2
    row_ind2, col_ind2 = linear_sum_assignment(D)
    num_match = (D[row_ind2, col_ind2]<1).sum()
    if num_match == 0:
        Recall = Precision = F1 = 0
    else:
        Recall = num_match/NGT
        Precision = num_match/NMask
        F1 = 2*Recall*Precision/(Recall+Precision)
    return Recall, Precision, F1


def refine_seperate(masks_final_2, times_final, cons=1, thresh_mask=0.5, ThreshJ=0.5):
    num_masks=len(times_final)
    if cons>1:
        kernel = np.ones(cons-1)
        times_cons=np.zeros(num_masks)
        for kk in range(num_masks):
            times_diff1=(np.diff(times_final[kk])==1)
            if len(times_diff1)>=cons-1:
                times_cons[kk]=sum(signal.convolve(times_diff1,kernel)==(cons-1))
    else:
        times_cons=np.array([x.size for x in times_final])

    if times_cons.sum()>0:
        masks_select_2=sparse.vstack([masks_final_2[x] for x in range(num_masks) if times_cons[x]>0])
        Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
    else:
        print('No consecutively active masks found. Please lower cons.')
        Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
    return Masks_2


def refine_seperate_nommin(masks_final_2, times_final, cons=1, thresh_mask=0.5):
    ThreshJ = 0.5
    num_masks=len(times_final)
    if cons>1:
        have_cons=np.zeros(num_masks, dtype='bool')
        for kk in range(num_masks):
            times_diff1 = times_final[kk][cons-1:] - times_final[kk][:1-cons]
            have_cons[kk] = np.any(times_diff1==cons-1)
        if np.any(have_cons):
            masks_select_2 = masks_final_2[have_cons]
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
        else:
            print('No masks found. Please lower cons.')
            Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
    else:
        masks_select_2 = masks_final_2
        Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')

    return Masks_2


def refine_seperate_nommin_float(masks_final_2, times_final, cons=1, thresh_mask=0.5):
    ThreshJ = 0.5
    num_masks=len(times_final)
    if cons>1:
        have_cons=np.zeros(num_masks, dtype='bool')
        for kk in range(num_masks):
            times_diff1 = times_final[kk][cons-1:] - times_final[kk][:1-cons]
            have_cons[kk] = np.any(times_diff1==cons-1)
        if np.any(have_cons):
            masks_select_2 = masks_final_2[have_cons]
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
        else:
            print('No masks found. Please lower cons.')
            masks_select_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
    else:
        masks_select_2 = masks_final_2
        # Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')

    return masks_select_2


def refine_seperate_multi(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask=0.5, display=False):
    ThreshJ = 0.5
    L_cons=len(list_cons)
    num_masks=len(times_final)
    Precision_k = np.zeros(L_cons)
    Recall_k = np.zeros(L_cons)
    F1_k = np.zeros(L_cons)
    for (k1, cons) in enumerate(list_cons):
        if cons>1:
            have_cons=np.zeros(num_masks, dtype='bool')
            for kk in range(num_masks):
                times_diff1 = times_final[kk][cons-1:] - times_final[kk][:1-cons]
                have_cons[kk] = np.any(times_diff1==cons-1)
            if np.any(have_cons):
                masks_select_2 = masks_final_2[have_cons]
                Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
                (Recall_k[k1], Precision_k[k1], F1_k[k1]) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ)
            else:
                (Recall_k[k1], Precision_k[k1], F1_k[k1]) = (0, 0, 0)
        else:
            masks_select_2 = masks_final_2
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
            (Recall_k[k1], Precision_k[k1], F1_k[k1]) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ)
        
    if display:
        ind = F1_k.argmax()
        print('Recall={:0.6f}, Precision={:0.6f}, F1={:0.6f}, cons={}.'.format(
            Recall_k[ind], Precision_k[ind], F1_k[ind], list_cons[ind]))
    return Recall_k, Precision_k, F1_k


def refine_seperate_multi_mmin(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask=0.5, display=False):
    ThreshJ = 0.5
    L_cons=len(list_cons)
    num_masks=len(times_final)
    Precision_k = np.zeros(L_cons)
    Recall_k = np.zeros(L_cons)
    F1_k = np.zeros(L_cons)
    for (k1, cons) in enumerate(list_cons):
        if cons>1:
            kernel = np.ones(cons-1)
            times_cons=np.zeros(num_masks)
            for kk in range(num_masks):
                times_diff1=(np.diff(times_final[kk])==1)
                if len(times_diff1)>=cons-1:
                    times_cons[kk]=sum(signal.convolve(times_diff1,kernel)==(cons-1))
        else:
            times_cons=np.array([x.size for x in times_final])
        
        if times_cons.sum()>0:
            masks_select_2=sparse.vstack([masks_final_2[x] for x in range(len(times_final)) if times_cons[x]>0])
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2]) #.astype('float')
            (Recall_k[k1], Precision_k[k1], F1_k[k1]) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ)
        else:
            (Recall_k[k1], Precision_k[k1], F1_k[k1]) = (0, 0, 0)
    if display:
        ind = F1_k.argmax()
        print('Recall={:0.6f}, Precision={:0.6f}, F1={:0.6f}, cons={}.'.format(
            Recall_k[ind], Precision_k[ind], F1_k[ind], list_cons[ind]))
    return Recall_k, Precision_k, F1_k

