import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from suns.PostProcessing.par3 import fastCOMdistance


def segs_results(segs: list):
    '''Pull segmented masks and their properties together from a list of segmentation results from each frame.
        The outputs are the segmented masks and some statistics 
        (areas, centers, whether they are from watershed, and the frames they are from)

    Inputs: 
        segs (list): A list of segmented masks with statistics from each frame.

    Outputs:
        totalmasks (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the neuron masks to be merged.
        neuronstate (1D numpy.array of bool, shape = (n,)): Indicators of whether a neuron is obtained without watershed.
        COMs (2D numpy.array of float, shape = (n,2)): COMs of the neurons.
        areas (1D numpy.array of uint32, shape = (n,)): Areas of the neurons. 
        probmapID (1D numpy.array of uint32, shape = (n,): indices of frames when the neuron is active. 
    '''
    totalmasks = sparse.vstack([x[0] for x in segs])
    neuronstate = np.hstack([x[1] for x in segs])
    COMs = np.vstack([x[2] for x in segs])
    areas = np.hstack([x[3] for x in segs])
    probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
    return totalmasks, neuronstate, COMs, areas, probmapID


def unique_neurons2_simp(totalmasks:sparse.csr_matrix, neuronstate:np.array, COMs:np.array, \
        areas:np.array, probmapID:np.array, minArea=0, thresh_COM0=0, useMP=True):
    '''Initally merge neurons with close COM (COM distance smaller than "thresh_COM0") by adding them together.
        The outputs are the merged masks "uniques" and the indices of frames when they are active "times".

    Inputs: 
        totalmasks (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the neuron masks to be merged.
        neuronstate (1D numpy.array of bool, shape = (n,)): Indicators of whether a neuron is obtained without watershed.
        COMs (2D numpy.array of float, shape = (n,2)): COMs of the neurons.
        areas (1D numpy.array of uint32, shape = (n,)): Areas of the neurons. 
        probmapID (1D numpy.array of uint32, shape = (n,): indices of frames when the neuron is active. 
        minArea (float or int, default to 0): Minimum neuron area. 
        thresh_COM0 (float or int, default to 0): Threshold of COM distance. 
            Masks have COM distance smaller than "thresh_COM0" are considered the same neuron and will be merged.
        useMP (bool, defaut to True): indicator of whether numba is used to speed up. 

    Outputs:
        uniques (sparse.csr_matrix): the neuron masks after merging. 
        times (list of 1D numpy.array): indices of frames when the neuron is active.
    '''
    if minArea>0: # screen out neurons with very small area
        area_select = (areas > minArea)
        totalmasks = totalmasks[area_select]
        neuronstate = neuronstate[area_select]
        COMs = COMs[area_select]
        areas = areas[area_select]
        probmapID = probmapID[area_select]

    maxN = neuronstate.size # Number of current masks
    if maxN>0: # False: # 
        # Only masks obtained without watershed can be neurons merged to
        coreneurons = neuronstate.nonzero()[0] 
        rows, cols = [], []

        cnt = 0
        keeplist = np.zeros(maxN, dtype='bool') 
        unusedneurons = np.ones(maxN, dtype='bool')
        for c in coreneurons:
            if neuronstate[c]:
                if useMP: 
                    r = np.zeros(COMs.shape[0])
                    fastCOMdistance(COMs, COMs[c], r)
                else:
                    r = pairwise_distances(COMs, COMs[c:c + 1]).squeeze() 
                # Find neighbors of a mask: the masks whose COM distance is smaller than "thresh_COM", including itself
                neighbors = np.logical_and(r <= thresh_COM0, unusedneurons).nonzero()[0]
                # those neighbors have been merged, so they will not be searched again
                neuronstate[neighbors] = False
                unusedneurons[neighbors] = False
                cols.append(neighbors)
                rows.append(c * np.ones(neighbors.size))
                # This neuron will be kept, and its neighbors will be merged to it.
                keeplist[c] = True
                cnt += 1

        ind_row = np.hstack(rows).astype('int') # indices of neurons merged to
        ind_col = np.hstack(cols).astype('int') # indices of neurons to be merged

        val = np.ones(ind_row.size)

        comb = sparse.csr_matrix((val, (ind_row, ind_col)), (maxN, maxN))
        comb = comb[keeplist]
        uniques = comb.dot(totalmasks) # Add merged neurons using sparse matrix multiplication
        times = [probmapID[comb[ii].toarray().astype('bool').squeeze()] for ii in range(cnt)]
        
    else:
        uniques = sparse.csc_matrix((0,totalmasks.shape[1]), dtype='bool')
        times = []

    return uniques, times


def group_neurons(uniques: sparse.csr_matrix, thresh_COM, thresh_mask, dims: tuple, times: list, useMP=True):
    '''Further merge neurons with close COM (COM distance smaller than "thresh_COM") by adding them together. 
        The COM threshold is larger than that used in "unique_neurons2_simp". 
        The outputs are the merged masks "uniqueout" and the indices of frames when they are active "uniquetimes".

    Inputs: 
        uniques (sparse.csr_matrix of float): the neuron masks to be merged.
        thresh_COM (float or int): Threshold of COM distance. 
            Masks have COM distance smaller than "thresh_COM" are considered the same neuron and will be merged.
        thresh_mask (float between 0 and 1): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        dims (tuple of int, shape = (2,)): the lateral shape of the image.
        times (list of 1D numpy.array): indices of frames when the neuron is active. 
        useMP (bool, defaut to True): indicator of whether numba is used to speed up. 

    Outputs:
        uniqueout (sparse.csr_matrix): the neuron masks after merging. 
        uniquetimes (list of 1D numpy.array): indices of frames when the neuron is active.
    '''
    N = uniques.shape[0] # Number of current masks
    uniques_thresh = sparse.vstack([x >= x.max() * thresh_mask for x in uniques]) # binary masks
    COMs = np.zeros((N, 2))
    # Calculate COM
    for nn in range(N):
        inds = uniques_thresh[nn].nonzero()[1]
        xxs = inds // (dims[1])
        yys = inds % (dims[1])
        COMs[nn] = [xxs.mean(), yys.mean()]

    uniquelist = np.ones(N, dtype='bool') # neurons to be merged
    keeplist = np.zeros(N, dtype='bool') # neurons merged to
    cnt = 0
    rows, cols = [], []
    for i in range(N):
        if uniquelist[i]:
            if useMP: 
                r = np.zeros(COMs.shape[0])
                fastCOMdistance(COMs, COMs[i], r)
            else:
                r = pairwise_distances(COMs, COMs[i:i + 1]).squeeze() 
            # Find neighbors of a mask: the masks whose COM distance is smaller than "thresh_COM", including itself
            neighbors = (r <= thresh_COM).nonzero()[0] 
            # those neighbors have been merged, so they will not be searched again
            uniquelist[neighbors] = False
            cols.append(neighbors)
            rows.append(i * np.ones(neighbors.size, dtype='int32'))
            keeplist[i] = True
            cnt += 1

    ind_row = np.hstack(rows).astype('uint32') # indices of neurons merged to
    ind_col = np.hstack(cols).astype('uint32') # indices of neurons to be merged

    # If a mask is merged into multiple masks, keep only the one with the smallest COM distance
    ind_delete = []
    for i in range(N):
        inds = (ind_col == i).nonzero()[0]
        if inds.size > 1:
            used = ind_row[inds]
            r2 = pairwise_distances(COMs[used], COMs[i:i + 1]).squeeze()
            keep = r2.argmin()
            inds = np.delete(inds, keep)
            ind_delete.append(inds)
    val = np.ones(ind_row.size)
    if len(ind_delete) > 0:
        ind_delete = np.hstack(ind_delete)
        val[ind_delete] = False

    comb = sparse.csr_matrix((val, (ind_row, ind_col)), (N, N))
    comb = comb[keeplist]
    uniqueout = comb.dot(uniques) # Add merged neurons using sparse matrix multiplication
    # merge active frame indices
    uniquetimes = [np.hstack([times[ind] for ind in comb[ii].toarray().nonzero()[1]]) for ii in range(cnt)]
    return uniqueout, uniquetimes


def piece_neurons_IOU(neuronmasks: sparse.csr_matrix, thresh_mask, thresh_IOU, times: list):
    '''Merge neurons with high IoU (IoU > thresh_IOU) by adding them together.
        The outputs are the merged masks "neuronmasks" and the indices of frames when they are active "uniquetimes".

    Inputs: 
        neuronmasks (sparse.csr_matrix of float): the neuron masks to be merged.
        thresh_mask (float between 0 and 1): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        thresh_IOU (float between 0 and 1): Threshold of IoU. 
            Masks have IoU higher than "thresh_IOU" are considered the same neuron and will be merged.
        times (list of 1D numpy.array): indices of frames when the neuron is active. 

    Outputs:
        neuronmasks (sparse.csr_matrix): the neuron masks after merging. 
        uniquetimes (list of 1D numpy.array): indices of frames when the neuron is active.
    '''
    uniquetimes = times.copy()
    N = neuronmasks.shape[0] # Number of current masks
    tempmasks = sparse.vstack([x >= x.max() * thresh_mask for x in neuronmasks]).astype('float') # binary masks
    # Calculate IoU
    area_i = tempmasks.dot(tempmasks.T).toarray()
    area = area_i.diagonal()
    area_i = area_i - np.diag(area)
    a1 = np.expand_dims(area, axis=1).repeat(N, axis=1)
    a2 = a1.T
    area_u = a1 + a2 - area_i
    IOU = np.triu(area_i) / area_u # IOU matrix is semmetric, so keeping a half is sufficient

    (x, y) = (IOU >= thresh_IOU).nonzero() # indices of neuron pairs with high IOU
    belongs = np.arange(N) # the indices of neurons merged to. Initially they are just themselves.
    for (xi, yi) in zip(x, y):
        xto = belongs[xi]
        yto = belongs[yi]
        if xto != yto: # merge xto and yto
            addto = min(xto, yto)
            addfrom = max(xto, yto)
            move = (belongs == addfrom) # merge all neurons previously merged to this neuron
            belongs[move] = addto # merge larger index to smaller index
            uniquetimes[addto] = np.hstack([uniquetimes[addfrom], uniquetimes[addto]]) # merge active time
            uniquetimes[addfrom] = np.array([], dtype='uint32')

    remains = np.unique(belongs) # indices of unique neurons after merging
    comb = sparse.csr_matrix((np.ones(N), (belongs, np.arange(N))), (N, N))
    comb = comb[remains]
    neuronmasks = comb.dot(neuronmasks) # Add merged neurons using sparse matrix multiplication
    uniquetimes = [uniquetimes[k] for k in remains]

    return neuronmasks, uniquetimes


def piece_neurons_consume(neuronmasks: sparse.csr_matrix, avgArea, thresh_mask, thresh_consume, times: list):
    '''Merge neurons with high consume (consume > thresh_consume) ratio. 
        If the larger neuron is larger than "avgArea", disgard it. Otherwise, add them together.
        The outputs are the merged masks "neuronmasks" and the indices of frames when they are active "uniquetimes".

    Inputs: 
        neuronmasks (sparse.csr_matrix of float): the neuron masks to be merged.
        avgArea (float or int): The typical neuron area (unit: pixels). When two masks are merged,
            if the area of the smaller mask is larger than "avgArea", the larger mask will be ignored.
        thresh_mask (float between 0 and 1): Threashold to binarize the real-number mask.
            values higher than "thresh_mask" times the maximum value are set to be True.
        thresh_consume (float between 0 and 1): Threshold of consume ratio. 
            Masks have consume ratio higher than "thresh_consume" are considered the same neuron and will be merged.
        times (list of 1D numpy.array): indices of frames when the neuron is active.

    Outputs:
        neuronmasks (sparse.csr_matrix): the neuron masks after merging. 
        uniquetimes (list of 1D numpy.array): indices of frames when the neuron is active.
    '''
    uniquetimes = times.copy()
    N = neuronmasks.shape[0] # Number of current masks
    tempmasks = sparse.vstack([x >= x.max() * thresh_mask for x in neuronmasks]).astype('float')
    # Calculate consume ratio
    area_i = tempmasks.dot(tempmasks.T).toarray()
    area = area_i.diagonal()
    area_i = area_i - np.diag(area)
    a2 = np.expand_dims(area, axis=0).repeat(N, axis=0)
    consume = area_i / a2

    belongs = np.arange(N) # the indices of neurons merged to. Initially they are just themselves.
    throw = 2 * N + 1 # used to indicate neurons that are ignored
    (x, y) = (consume >= thresh_consume).nonzero() # indices of neuron pairs with consume
    for (xi, yi) in zip(x, y):
        area_x = area[xi]
        area_y = area[yi]
        if max(area_x, area_y) > avgArea: # If the larger neuron is larger than avgArea, throw it
            if area_y > area_x:
                belongs[yi] = throw
                uniquetimes[yi] = np.array([], dtype='uint32')
            else:
                belongs[xi] = throw
                uniquetimes[xi] = np.array([], dtype='uint32')
        else: # If both neurons are is smaller than avgArea, add them
            xto = belongs[xi]
            yto = belongs[yi]
            if xto != yto: # merge xto and yto
                addto = min(xto, yto)
                addfrom = max(xto, yto)
                if addfrom < throw: # if one of them is already thrown, do nothing
                    move = (belongs == addfrom)
                    belongs[move] = addto # merge larger index to smaller index
                    uniquetimes[addto] = np.hstack([uniquetimes[addfrom], uniquetimes[addto]]) # merge active time
                    uniquetimes[addfrom] = np.array([], dtype='uint32')

    keep = (belongs != throw).nonzero()[0] # indices of unique neurons after merging
    comb = sparse.csr_matrix((np.ones(keep.size), (belongs[keep], np.arange(N)[keep])), (N, N))
    remains = np.setdiff1d(belongs, throw)
    comb = comb[remains]
    neuronmasks = comb.dot(neuronmasks) # Add merged neurons using sparse matrix multiplication
    uniquetimes = [uniquetimes[k] for k in remains]

    return neuronmasks, uniquetimes

