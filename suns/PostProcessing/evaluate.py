import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
import time


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

