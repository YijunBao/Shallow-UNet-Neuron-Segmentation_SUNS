import io
import numpy as np
from scipy import sparse
from scipy import signal
from scipy.optimize import linear_sum_assignment
import time
import multiprocessing as mp


##
def GetPerformance_Jaccard_2(GTMasks, Masks, ThreshJ=0.5):
    '''Calculate the recall, precision, and F1 score of segmented neurons by comparing with ground truth.

    Inputs: 
        GTMasks (sparse.csr_matrix): Ground truth masks.
        Masks (sparse.csr_matrix): Segmented masks.
        ThreshJ (float, default to 0.5): Threshold Jaccard distance for two neurons to match.

    Outputs:
        Recall (float): Percentage of matched neurons over all GT neurons. 
        Precision (float): Percentage of matched neurons over all segmented neurons. 
        F1 (float): Harmonic mean of Recall and Precision. 
    '''
    if 'bool' in str(Masks.dtype): # bool cannot be used to calculate IoU
        Masks = Masks.astype('uint32')
    if 'bool' in str(GTMasks.dtype):
        GTMasks = GTMasks.astype('uint32')
    NGT = GTMasks.shape[0] # Number of GT neurons
    NMask = Masks.shape[0] # Number of segmented neurons
    a1 = np.repeat(GTMasks.sum(axis=1).A, NMask, axis=1)
    a2 = np.repeat(Masks.sum(axis=1).A.T, NGT, axis=0)
    intersectMat = GTMasks.dot(Masks.transpose()).A
    unionMat = a1 + a2 - intersectMat
    JaccardInd = intersectMat/unionMat # IoU between each pair of neurons
    Dmat = 1-JaccardInd # Jaccard distance is 1 - IoU
    # Dmat[intersectMat == a1] = 0
    # Dmat[intersectMat == a2] = 0
    D = Dmat
    # When Jaccard distance is larger than ThreshJ, it is set to 2, meaning infinity
    D[D > ThreshJ] = 2 
    # Use Hungarian algorithm to match two sets of neurons
    row_ind2, col_ind2 = linear_sum_assignment(D) 
    num_match = (D[row_ind2, col_ind2]<1).sum() # Number of matched neurons
    if num_match == 0:
        Recall = Precision = F1 = 0
    else:
        Recall = num_match/NGT
        Precision = num_match/NMask
        F1 = 2*Recall*Precision/(Recall+Precision)
    return Recall, Precision, F1


def refine_seperate(masks_final_2, times_final, cons=1, thresh_mask=0.5, ThreshJ=0.5):
    '''Refine segmented neurons by requiring them to be active for "cons" consecutive frames.

    Inputs: 
        masks_final_2 (sparse.csr_matrix of float32): the segmented neuron masks. 
        times_final (list of 1D numpy.array): indecis of frames when the neuron is active.
        cons (int, default to 1): Minimum number of consecutive frames that a neuron should be active for.
        thresh_mask (float between 0 and 1, default to 0.5): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        ThreshJ (float, default to 0.5): Threshold Jaccard distance for two neurons to match.

    Outputs:
        Masks_2 (sparse.csr_matrix of bool): the final segmented binary neuron masks after consecutive refinement. 
    '''
    num_masks=len(times_final)
    if num_masks:
        if cons>1:
            have_cons=np.zeros(num_masks, dtype='bool')
            for kk in range(num_masks):
                times_diff1 = times_final[kk][cons-1:] - times_final[kk][:1-cons]
                # indicators of whether the neuron was active for "cons" frames
                have_cons[kk] = np.any(times_diff1==cons-1) 
            if np.any(have_cons):
                masks_select_2 = masks_final_2[have_cons]
                Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2])
            else:
                print('No masks found. Please lower cons.')
                Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
        else:
            masks_select_2 = masks_final_2
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2])
    else:
        Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')

    return Masks_2


def refine_seperate_multi(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask=0.5, ThreshJ = 0.5, display=False):
    '''Refine segmented neurons by requiring them to be active for "cons" consecutive frames.
        Used to search the optimal number of consecutive frames.

    Inputs: 
        GTMasks_2 (sparse.csr_matrix): Ground truth masks.
        masks_final_2 (sparse.csr_matrix of float32): the segmented neuron masks. 
        times_final (list of 1D numpy.array): indecis of frames when the neuron is active.
        list_cons (list of int): A list of minimum number of consecutive frames that a neuron should be active for.
        thresh_mask (float between 0 and 1, default to 0.5): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        ThreshJ (float, default to 0.5): Threshold Jaccard distance for two neurons to match.
        display (bool, default to False): Indicator of whether to show the optimal "cons"

    Outputs:
        Recall_k (1D numpy.array of float): Percentage of matched neurons over all GT neurons. 
        Precision_k (1D numpy.array of float): Percentage of matched neurons over all segmented neurons. 
        F1_k (1D numpy.array of float): Harmonic mean of Recall and Precision. 
    '''
    L_cons=len(list_cons) # number of "cons" to optimize over
    num_masks=len(times_final) # Number of segmented neurons
    Precision_k = np.zeros(L_cons)
    Recall_k = np.zeros(L_cons)
    F1_k = np.zeros(L_cons)
    for (k1, cons) in enumerate(list_cons):
        if cons>1:
            have_cons=np.zeros(num_masks, dtype='bool')
            for kk in range(num_masks):
                times_diff1 = times_final[kk][cons-1:] - times_final[kk][:1-cons]
                # indicators of whether the neuron was active for "cons" frames
                have_cons[kk] = np.any(times_diff1==cons-1)
            if np.any(have_cons):
                masks_select_2 = masks_final_2[have_cons]
                Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2])
                # Evalueate the accuracy of the result using recall, precision, and F1
                (Recall_k[k1], Precision_k[k1], F1_k[k1]) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ)
            else:
                (Recall_k[k1], Precision_k[k1], F1_k[k1]) = (0, 0, 0)
        else:
            masks_select_2 = masks_final_2
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2])
            (Recall_k[k1], Precision_k[k1], F1_k[k1]) = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,ThreshJ)
        
    if display:
        ind = F1_k.argmax()
        print('Recall={:0.6f}, Precision={:0.6f}, F1={:0.6f}, cons={}.'.format(
            Recall_k[ind], Precision_k[ind], F1_k[ind], list_cons[ind]))
    return Recall_k, Precision_k, F1_k

