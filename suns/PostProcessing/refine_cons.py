import numpy as np
from scipy import sparse
import time

from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2


def refine_seperate(masks_final_2, times_final, cons=1, thresh_mask=0.5, ThreshJ=0.5):
    '''Refine segmented neurons by requiring them to be active for "cons" consecutive frames.
        The output are "Masks_2", a 2D sparse matrix of the final segmented neurons, 
        and "times_cons", a list of indices of frames when the final neuron is active.

    Inputs: 
        masks_final_2 (sparse.csr_matrix of float32): the segmented neuron masks. 
        times_final (list of 1D numpy.array): indices of frames when the neuron is active.
        cons (int, default to 1): Minimum number of consecutive frames that a neuron should be active for.
        thresh_mask (float between 0 and 1, default to 0.5): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        ThreshJ (float between 0 and 1, default to 0.5): Threshold Jaccard distance for two neurons to match.

    Outputs:
        Masks_2 (sparse.csr_matrix of bool): the final segmented binary neuron masks after consecutive refinement. 
        times_cons (list of 1D numpy.array): indices of frames when the final neuron is active.
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
                times_cons = [x for (c,x) in zip(have_cons,times_final) if c]
            else:
                print('No masks found. Please lower cons.')
                Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
                times_cons = []
        else:
            masks_select_2 = masks_final_2
            Masks_2 = sparse.vstack([x >= x.max() * thresh_mask for x in masks_select_2])
            times_cons = times_final
    else:
        Masks_2 = sparse.csc_matrix((0,masks_final_2.shape[1]), dtype='bool')
        times_cons = times_final

    return Masks_2, times_cons


def refine_seperate_multi(GTMasks_2, masks_final_2, times_final, list_cons, thresh_mask=0.5, ThreshJ = 0.5, display=False):
    '''Refine segmented neurons by requiring them to be active for "cons" consecutive frames.
        The outputs are the recall, precision, and F1 calculated using all values in "list_cons".
        Used to search the optimal "cons".

    Inputs: 
        GTMasks_2 (sparse.csr_matrix): Ground truth masks.
        masks_final_2 (sparse.csr_matrix of float32): the segmented neuron masks. 
        times_final (list of 1D numpy.array): indices of frames when the neuron is active.
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

