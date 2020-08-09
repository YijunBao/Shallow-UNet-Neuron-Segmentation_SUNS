import numpy as np
import cv2
from scipy import sparse
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max


def watershed_ski(seg: bool): # Watershed using skimage
    dist_transform = cv2.distanceTransform(seg, cv2.DIST_L2, 0) # 0
    local_maxi = peak_local_max(dist_transform, indices=False, footprint=np.ones((3, 3)), labels=seg)
    markers = ndi.label(local_maxi)[0]  # ?
    wt = watershed(-dist_transform, markers, mask=seg, watershed_line=True, connectivity=2).astype('uint8')  #
    return wt


def watershed_ski_nomarker(seg: bool): # Watershed using skimage
    dist_transform = cv2.distanceTransform(seg, cv2.DIST_L2, 0) # 0
    wt = watershed(-dist_transform, mask=seg, watershed_line=True, connectivity=2).astype('uint8')  #
    return wt


def watershed_CV(seg: bool): # Watershed using OpenCV
    bw = 255 * seg.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = 255 - bw
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3) # 0
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=1).astype('uint8')
    m = sure_bg + sure_fg
    _, markers = cv2.connectedComponents(m.astype('uint8'))
    mm = cv2.watershed(cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB), markers)
    bw[mm == -1] = 0
    return bw

def watershed_CV_dilate(seg: bool): # Watershed using OpenCV
    bw = 255 * seg.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = 255 - cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3) # 0
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=1).astype('uint8')
    m = sure_bg + sure_fg
    _, markers = cv2.connectedComponents(m.astype('uint8'))
    mm = cv2.watershed(cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB), markers)
    bw[mm == -1] = 0
    return bw

def watershed_CV_dilate_unknown(seg: bool): # Watershed using OpenCV
    bw = 255 * seg.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3) # 0
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=1).astype('uint8')
    unknown = cv2.subtract(sure_bg, sure_fg)    
    _, markers = cv2.connectedComponents(sure_fg, connectivity=4)    
    markers = markers + 1    
    markers[unknown == 255] = 0
    mm = cv2.watershed(cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB), markers)
    bw[mm == -1] = 0
    return bw

def watershed_CV_unknown(seg: bool): # Watershed using OpenCV
    bw = 255 * seg.astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = bw
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3) # 0
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=1).astype('uint8')
    unknown = cv2.subtract(sure_bg, sure_fg)    
    _, markers = cv2.connectedComponents(sure_fg, connectivity=4)    
    markers = markers + 1    
    markers[unknown == 255] = 0
    mm = cv2.watershed(cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB), markers)
    bw[mm == -1] = 0
    return bw


def refineNeuron_WT(seg: bool, minArea, avgArea):
    # wt = watershed_ski(seg)
    # wt = watershed_ski_nomarker(seg)
    # wt = watershed_CV(seg)
    # wt = watershed_CV_dilate(seg)
    wt = watershed_CV_dilate_unknown(seg)
    # wt = watershed_CV_unknown(seg)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(wt, connectivity=4) #
    tempstate = (nlabels <= 2)    
    totalx, totaly, tempcent, temparea = [], [], [], []
    #    neuron_cnt = []
    for k in range(1, nlabels):
        area_k = stats[k][4]
        # if area_k > minArea:
        if (minArea < area_k <= avgArea) or (area_k > avgArea and tempstate):
            neuronSegment = (labels == k)
            tempy, tempx = neuronSegment.nonzero()
            totalx.append(tempx)
            totaly.append(tempy)
            tempcent.append(centroids[k, :])
            temparea.append(area_k)
        elif (area_k > avgArea) and not tempstate:
            neuronSegment = (labels == k)
            tempxl, tempyl, centsl, _, areal = refineNeuron_WT(neuronSegment.astype('uint8'), minArea, avgArea) # , eng
            dcnt = len(centsl)
            if dcnt:
                totalx = totalx + tempxl
                totaly = totaly + tempyl
                tempcent = tempcent + centsl
                temparea = temparea + areal

    return totalx, totaly, tempcent, tempstate, temparea


def refineNeuron_WT_keep(seg: bool, minArea, avgArea):
    # wt = watershed_ski(seg)
    # wt = watershed_ski_nomarker(seg)
    # wt = watershed_CV(seg)
    # wt = watershed_CV_dilate(seg)
    wt = watershed_CV_dilate_unknown(seg)
    # wt = watershed_CV_unknown(seg)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(wt, connectivity=4) #
    tempstate = (nlabels <= 2)    
    totalx, totaly, tempcent, temparea = [], [], [], []
    #    neuron_cnt = []
    for k in range(1, nlabels):
        area_k = stats[k][4]
        # if area_k > minArea:
        if (minArea < area_k <= avgArea) or (area_k > avgArea and tempstate):
            neuronSegment = (labels == k)
            tempy, tempx = neuronSegment.nonzero()
            totalx.append(tempx)
            totaly.append(tempy)
            tempcent.append(centroids[k, :])
            temparea.append(area_k)
        elif (area_k > avgArea) and not tempstate:
            neuronSegment = (labels == k)
            tempxl, tempyl, centsl, _, areal = refineNeuron_WT_keep(neuronSegment.astype('uint8'), minArea, avgArea) # , eng
            dcnt = len(centsl)
            if dcnt:
                totalx = totalx + tempxl
                totaly = totaly + tempyl
                tempcent = tempcent + centsl
                temparea = temparea + areal

    if not tempcent: # if all the cutted neurons are too small, then cancel watershed.
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=4) #
        # neuronSegment = (labels == 1)
        tempy, tempx = labels.nonzero()
        totalx.append(tempx)
        totaly.append(tempy)
        tempcent.append(centroids[1, :])
        temparea.append(stats[1][4])
        tempstate = True

    return totalx, totaly, tempcent, tempstate, temparea
    

def watershed_neurons(dims, frame_seg, minArea, avgArea):
    '''Try to further segment large masks using watershed.

    Inputs: 
        dims (tuple of int, shape = (2,)): the lateral shape of the region.
        frame_seg (a list of 4 elements): corresponding to the four outputs of this function, but before watershed.
        minArea (float or int, default to 0): Minimum area of a valid neuron mask (unit: pixels).
        avgArea (float or int, default to 0): The typical neuron area (unit: pixels). 
            Neuron masks with areas larger than avgArea will be further segmented by watershed.

    Outputs:
        masks (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the segmented neuron masks.
            Totally "n" neurons. Each neuron is represented by a 1D array reshaped from a 2D binary image.
            Nonzero points belong to the neuron. 
        neuronstate (1D numpy.ndarray of bool, shape = (n,)): Indicators of whether a neuron is obtained without watershed. 
        cents (2D numpy.ndarray of float, shape = (n,2)): COMs of the neurons.
        areas (1D numpy.ndarray of int, shape = (n,)): Areas of the neurons.
    '''
    masks = frame_seg[0]
    neuronstate = frame_seg[1]
    cents = frame_seg[2]
    areas = frame_seg[3]
    num = areas.size
    if num>0:
        masks_new, neuronstate_new, cents_new, areas_new = [], [], [], []
        for (k, k_area) in enumerate(areas):
            if k_area > avgArea:
                # Crop a small rectangle containing mask k
                _, inds = masks[k].nonzero()
                rows = inds // dims[1]
                cols = inds % dims[1]
                xmin = cols.min() - 1
                ymin = rows.min() - 1
                xsize = cols.max()-cols.min()+3
                ysize = rows.max()-rows.min()+3
                BW1 = sparse.coo_matrix((np.ones(k_area, dtype = 'uint8'), (rows-ymin, cols-xmin)), shape = (ysize, xsize)).toarray()
                
                # Apply watershed to each mask, and create a series of possibly smaller masks
                tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT(BW1, minArea, avgArea)
                # tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT_keep(BW1, minArea, avgArea)
                dcnt = len(tempcent)
                if dcnt > 0:
                    # convert a 2D coordiate into a 1D index
                    # weighted by 1/4 for masks segmented after watershed 
                    masks_new = masks_new + [sparse.csr_matrix((np.ones(x1.size)*(1 / 4 + 3 / 4 * tempstate), \
                        (np.zeros(x1.size, dtype='int'), (y1 + ymin) * dims[1] + (x1 + xmin))), \
                        shape = (1, dims[0]*dims[1])) for (x1, y1) in zip(tempx1, tempy1)]
                    neuronstate_new = neuronstate_new + [tempstate] * dcnt
                    cents_new = cents_new + [tj + [xmin, ymin] for tj in tempcent]
                    areas_new = areas_new + temparea

            else:
                # add the information of the segmented mask to the output lists
                masks_new.append(masks[k])
                neuronstate_new.append(True)
                cents_new.append(cents[k])
                areas_new.append(k_area)

        neuron_cnt = len(masks_new)
        if neuron_cnt == 0: # does not find any neuron
            masks = sparse.csr_matrix((0, dims[0] * dims[1]))
            neuronstate = np.array([], dtype='int')
            cents = np.empty((0, 2))
            areas = np.array([], dtype='int')
        else:
            # convert "masks" from list to sparse matrix
            masks = sparse.vstack(masks_new)
            # convert "neuronstate", "cents", and "areas" from list to numpy.array
            neuronstate = np.array(neuronstate_new) 
            cents = np.array(cents_new)
            areas = np.array(areas_new)

    return masks, neuronstate, cents, areas
    

def separateNeuron(img: np.array, thresh_pmap=None, minArea=0, avgArea=0, useWT=False):
    '''Segment a frame of image (probablity map) into active neuron masks.

    Inputs: 
        img (2D numpy.ndarray of bool, uint8, uint16, int16, float32, or float64): the probablity map to be segmented.
        thresh_pmap (float or int, default to None): The probablity threshold. Values higher than thresh_pmap are active pixels. 
            if thresh_pmap==None, then thresholding is not performed. This is used when thresholding is done before this function.
        minArea (float or int, default to 0): Minimum area of a valid neuron mask (unit: pixels).
        avgArea (float or int, default to 0): The typical neuron area (unit: pixels). If watershed is used, 
            neuron masks with areas larger than avgArea will be further segmented by watershed.
        useWT (bool, default to False): Indicator of whether watershed is used. 

    Outputs:
        masks (sparse.csr_matrix of float32, shape = (n,Lx*Ly)): the segmented neuron masks.
            Totally "n" neurons. Each neuron is represented by a 1D array reshaped from a 2D binary image.
            Nonzero points belong to the neuron. 
        neuronstate (1D numpy.ndarray of bool, shape = (n,)): Indicators of whether a neuron is obtained without watershed. 
            if useWT=False, then all the elements should be True.
        cents (2D numpy.ndarray of float, shape = (n,2)): COMs of the neurons.
        areas (1D numpy.ndarray of int, shape = (n,)): Areas of the neurons.
    '''
    dims = img.shape
    if img.dtype =='bool' or thresh_pmap is None: # skip thresholding
        thresh1 = img
    else: # Threshold the input image to binary
        _, thresh1 = cv2.threshold(img, thresh_pmap, 255, cv2.THRESH_BINARY)
    thresh1 = thresh1.astype('uint8') # Convert the binary image to uint8

    # Segment the binary image into connected components with statistics
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=4)

    col = [] # will store non-zero elements of each segmented neuron
    neuronstate, cents, areas = [], [], []
    if nlabels > 0: # nlabel is the number of connected components. If nlabel=0, no active pixel is found
        for k in range(1, nlabels): # for all connected components 
            current_stat = stats[k]
            k_area = current_stat[4]
            if k_area > minArea: # Only keep the connected regions with area larger than minArea
                BW = (labels == k)
                # Crop a small rectangle containing mask k
                xmin = max((0, current_stat[0] - 1))
                xmax = min((dims[1], current_stat[0] + current_stat[2] + 1))
                ymin = max((0, current_stat[1] - 1))
                ymax = min((dims[0], current_stat[1] + current_stat[3] + 1))
                BW1 = BW[ymin:ymax, xmin:xmax]

                if useWT and k_area > avgArea: # If useWT==True and the mask area is larger than avgArea, 
                    # then try to use watershed to further segment the mask
                    # Zero-pad the rectangular regions if necessary, so that no active pixels are on the boundarys.
                    if current_stat[0]==0:
                        xmin = -1
                        BW1 = np.pad(BW1, ((0,0),(1,0)),'constant', constant_values=(0, 0))
                    if current_stat[0] + current_stat[2]==dims[1]:
                        BW1 = np.pad(BW1, ((0,0),(0,1)),'constant', constant_values=(0, 0))
                    if current_stat[1]==0:
                        ymin = -1
                        BW1 = np.pad(BW1, ((1,0),(0,0)),'constant', constant_values=(0, 0))
                    if current_stat[1] + current_stat[3]==dims[0]:
                        BW1 = np.pad(BW1, ((0,1),(0,0)),'constant', constant_values=(0, 0))
                    tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT(BW1.astype('uint8'), minArea, avgArea)
                    # tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT_keep(BW1.astype('uint8'), minArea, avgArea)
                    
                    dcnt = len(tempcent) # Number of segmented pieces obtained by watershed
                    if dcnt > 0:
                        # convert a 2D coordiate into a 1D index
                        tempind = [(y1 + ymin) * dims[1] + (x1 + xmin) for (x1, y1) in zip(tempx1, tempy1)]
                        col = col + tempind
                        neuronstate = neuronstate + [tempstate] * dcnt
                        cents = cents + [tj + [xmin, ymin] for tj in tempcent]
                        areas = areas + temparea

                else: # If not using watershed, each mask is final
                    tempy, tempx = BW1.nonzero()
                    # convert a 2D coordiate into a 1D index
                    tempind = (tempy + ymin) * dims[1] + (tempx + xmin)
                    # add the information of the segmented mask to the output lists
                    col.append(tempind)
                    neuronstate.append(True)
                    cents.append(centroids[k])
                    areas.append(k_area)

    neuron_cnt = len(col)
    if neuron_cnt == 0: # does not find any neuron
        masks = sparse.csr_matrix((neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array([], dtype='int')
        cents = np.empty((0, 2))
        areas = np.array([], dtype='int')
    else:
        ind_col = np.hstack(col)
        temp = [np.ones(x.size, dtype='int') for x in col]
        ind_row = np.hstack([j * tj for (j, tj) in enumerate(temp)]) # corresponding index of segmented neurons
        # weighted by 1/4 for masks segmented after watershed 
        vals = np.hstack([(1 / 4 + 3 / 4 * sj) * tj for (sj, tj) in zip(neuronstate, temp)])
        # create a sparse matrix with "neuron_cnt" rows to represent the "neuron_cnt" segmented neurons
        masks = sparse.csr_matrix((vals, (ind_row, ind_col)), shape=(neuron_cnt, dims[0] * dims[1]))
        # convert "neuronstate", "cents", and "areas" from list to numpy.array
        neuronstate = np.array(neuronstate) 
        cents = np.array(cents)
        areas = np.array(areas)

    return masks, neuronstate, cents, areas
    