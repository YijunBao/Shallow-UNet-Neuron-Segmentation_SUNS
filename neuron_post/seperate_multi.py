import numpy as np
import cv2
from scipy import sparse
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
# import matlab
# import matlab.engine as engine


# def watershed_ski_matlab(seg: bool): # Watershed using matlab
#     dist_transform = cv2.distanceTransform(seg, cv2.DIST_L2, 0) # 0
#     wt = np.array(eng.watershed(matlab.double((-dist_transform).tolist()))) * seg
#     return wt


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


def refineNeuron_WT(seg: bool, minArea, avgArea): #, eng=None
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


def refineNeuron_WT_keep(seg: bool, minArea, avgArea): #, eng=None
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
    

def separateNeuron_noWT(img: np.array, thresh_pmap, minArea): #, eng=None
    dims = img.shape
    if img.dtype =='bool':
        thresh1 = img
    else:
        _, thresh1 = cv2.threshold(img, thresh_pmap, 255, cv2.THRESH_BINARY) #255 * 
    thresh1 = thresh1.astype('uint8')
    # kernel = np.ones((3, 3), np.uint8)
    # thresh1 = cv2.erode(thresh1, kernel, iterations=1)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=4)
    col = []
    neuronstate, cents, areas = [], [], []
    if nlabels > 0:
        for k in range(1, nlabels):
            current_stat = stats[k]
            k_area = current_stat[4]
            if k_area > minArea:
                BW = (labels == k)
                xmin = max((0, current_stat[0] - 1))
                xmax = min((dims[1], current_stat[0] + current_stat[2] + 1))
                ymin = max((0, current_stat[1] - 1))
                ymax = min((dims[0], current_stat[1] + current_stat[3] + 1))
                # the -1 and +1 in the above four line can be removed for a faster speed
                BW1 = BW[ymin:ymax, xmin:xmax]

                tempy, tempx = BW1.nonzero()
                tempind = (tempy + ymin) * dims[1] + (tempx + xmin)
                col.append(tempind)
                neuronstate.append(True)
                cents.append(centroids[k])
                areas.append(k_area)

    neuron_cnt = len(col)
    if neuron_cnt == 0:
        masks = sparse.csr_matrix((neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array([], dtype='int')
        cents = np.empty((0, 2))
        areas = np.array([], dtype='int')
    else:
        ind_col = np.hstack(col)
        temp = [np.ones(x.size, dtype='int') for x in col]
        ind_row = np.hstack([j * tj for (j, tj) in enumerate(temp)])
        vals = np.hstack(temp)
        masks = sparse.csr_matrix((vals, (ind_row, ind_col)), shape=(neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array(neuronstate) 
        cents = np.array(cents)
        areas = np.array(areas)

    return masks, neuronstate, cents, areas


def watershed_neurons(dims, frame_seg, minArea, avgArea):
    masks = frame_seg[0]
    neuronstate = frame_seg[1]
    cents = frame_seg[2]
    areas = frame_seg[3]
    num = areas.size
    if num>0:
        masks_new, neuronstate_new, cents_new, areas_new = [], [], [], []
        for (k, k_area) in enumerate(areas):
            if k_area > avgArea:
                _, inds = masks[k].nonzero()
                rows = inds // dims[1]
                cols = inds % dims[1]
                # xmin = max((0, cols.min() - 1))
                # ymin = max((0, rows.min() - 1))
                xmin = cols.min() - 1
                ymin = rows.min() - 1
                xsize = cols.max()-cols.min()+3
                ysize = rows.max()-rows.min()+3
                BW1 = sparse.coo_matrix((np.ones(k_area, dtype = 'uint8'), (rows-ymin, cols-xmin)), shape = (ysize, xsize)).todense().A
                
                tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT(BW1.astype('uint8'), minArea, avgArea) #, eng
                # tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT_keep(BW1.astype('uint8'), minArea, avgArea) #, eng
                dcnt = len(tempcent)
                if dcnt > 0:
                    masks_new = masks_new + [sparse.csr_matrix((np.ones(x1.size)*(1 / 4 + 3 / 4 * tempstate), (np.zeros(x1.size, dtype='int'), \
                        (y1 + ymin) * dims[1] + (x1 + xmin))), shape = (1, dims[0]*dims[1])) for (x1, y1) in zip(tempx1, tempy1)]
                    neuronstate_new = neuronstate_new + [tempstate] * dcnt
                    cents_new = cents_new + [tj + [xmin, ymin] for tj in tempcent]
                    areas_new = areas_new + temparea

            else:
                masks_new.append(masks[k])
                neuronstate_new.append(True)
                cents_new.append(cents[k])
                areas_new.append(k_area)

        neuron_cnt = len(masks_new)
        if neuron_cnt == 0:
            masks = sparse.csr_matrix((0, dims[0] * dims[1]))
            neuronstate = np.array([], dtype='int')
            cents = np.empty((0, 2))
            areas = np.array([], dtype='int')
        else:
            masks = sparse.vstack(masks_new)
            neuronstate = np.array(neuronstate_new) 
            cents = np.array(cents_new)
            areas = np.array(areas_new)

    return masks, neuronstate, cents, areas
    

def separateNeuron(img: np.array, thresh_pmap, minArea, avgArea, useWT=False): #, eng=None
    dims = img.shape
    if img.dtype =='bool':
        thresh1 = img
    else:
        _, thresh1 = cv2.threshold(img, thresh_pmap, 255, cv2.THRESH_BINARY) #255 * 
    thresh1 = thresh1.astype('uint8')
    # kernel = np.ones((3, 3), np.uint8)
    # thresh1 = cv2.erode(thresh1, kernel, iterations=1)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=4)
    col = []
    neuronstate, cents, areas = [], [], []
    if nlabels > 0:
        for k in range(1, nlabels):
            current_stat = stats[k]
            k_area = current_stat[4]
            if k_area > minArea:
                BW = (labels == k)
                xmin = max((0, current_stat[0] - 1))
                xmax = min((dims[1], current_stat[0] + current_stat[2] + 1))
                ymin = max((0, current_stat[1] - 1))
                ymax = min((dims[0], current_stat[1] + current_stat[3] + 1))
                BW1 = BW[ymin:ymax, xmin:xmax]

                if useWT and k_area > avgArea: # False: # 
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
                    tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT(BW1.astype('uint8'), minArea, avgArea) #, eng
                    # tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT_keep(BW1.astype('uint8'), minArea, avgArea) #, eng
                    
                    dcnt = len(tempcent)
                    if dcnt > 0:
                        tempind = [(y1 + ymin) * dims[1] + (x1 + xmin) for (x1, y1) in zip(tempx1, tempy1)]
                        col = col + tempind
                        neuronstate = neuronstate + [tempstate] * dcnt
                        cents = cents + [tj + [xmin, ymin] for tj in tempcent]
                        areas = areas + temparea

                else:
                    tempy, tempx = BW1.nonzero()
                    tempind = (tempy + ymin) * dims[1] + (tempx + xmin)
                    col.append(tempind)
                    neuronstate.append(True)
                    cents.append(centroids[k])
                    areas.append(k_area)

    neuron_cnt = len(col)
    if neuron_cnt == 0:
        masks = sparse.csr_matrix((neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array([], dtype='int')
        cents = np.empty((0, 2))
        areas = np.array([], dtype='int')
    else:
        ind_col = np.hstack(col)
        temp = [np.ones(x.size, dtype='int') for x in col]
        ind_row = np.hstack([j * tj for (j, tj) in enumerate(temp)])
        vals = np.hstack([(1 / 4 + 3 / 4 * sj) * tj for (sj, tj) in zip(neuronstate, temp)])
        masks = sparse.csr_matrix((vals, (ind_row, ind_col)), shape=(neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array(neuronstate) 
        cents = np.array(cents)
        areas = np.array(areas)

    return masks, neuronstate, cents, areas
    

def separateNeuron_b(img: np.array('uint8'), thresh_pmap, minArea, avgArea, useWT=False): #, eng=None
    '''pre-thresholded (binary) probability map'''
    dims = img.shape
    # kernel = np.ones((3, 3), np.uint8)
    # thresh1 = cv2.erode(thresh1, kernel, iterations=1)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    col = []
    neuronstate, cents, areas = [], [], []
    if nlabels > 0:
        for k in range(1, nlabels):
            current_stat = stats[k]
            k_area = current_stat[4]
            if k_area > minArea:
                BW = (labels == k)
                xmin = max((0, current_stat[0] - 1))
                xmax = min((dims[1], current_stat[0] + current_stat[2] + 1))
                ymin = max((0, current_stat[1] - 1))
                ymax = min((dims[0], current_stat[1] + current_stat[3] + 1))
                BW1 = BW[ymin:ymax, xmin:xmax]

                if useWT and k_area > avgArea: # False: # 
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
                    tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT(BW1.astype('uint8'), minArea, avgArea) #, eng
                    # tempx1, tempy1, tempcent, tempstate, temparea = refineNeuron_WT_keep(BW1.astype('uint8'), minArea, avgArea) #, eng
                    
                    dcnt = len(tempcent)
                    if dcnt > 0:
                        tempind = [(y1 + ymin) * dims[1] + (x1 + xmin) for (x1, y1) in zip(tempx1, tempy1)]
                        col = col + tempind
                        neuronstate = neuronstate + [tempstate] * dcnt
                        cents = cents + [tj + [xmin, ymin] for tj in tempcent]
                        areas = areas + temparea

                else:
                    tempy, tempx = BW1.nonzero()
                    tempind = (tempy + ymin) * dims[1] + (tempx + xmin)
                    col.append(tempind)
                    neuronstate.append(True)
                    cents.append(centroids[k])
                    areas.append(k_area)

    neuron_cnt = len(col)
    if neuron_cnt == 0:
        masks = sparse.csr_matrix((neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array([], dtype='int')
        cents = np.empty((0, 2))
        areas = np.array([], dtype='int')
    else:
        ind_col = np.hstack(col)
        temp = [np.ones(x.size, dtype='int') for x in col]
        ind_row = np.hstack([j * tj for (j, tj) in enumerate(temp)])
        vals = np.hstack([(1 / 4 + 3 / 4 * sj) * tj for (sj, tj) in zip(neuronstate, temp)])
        masks = sparse.csr_matrix((vals, (ind_row, ind_col)), shape=(neuron_cnt, dims[0] * dims[1]))
        neuronstate = np.array(neuronstate) 
        cents = np.array(cents)
        areas = np.array(areas)

    return masks, neuronstate, cents, areas
    