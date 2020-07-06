import numpy as np
from scipy import sparse
import multiprocessing as mp
from sklearn.metrics import pairwise_distances
from par3 import fastCOMdistance
# import matlab
# import matlab.engine as engine


var_dict = {}
def init_pool(mp_COMs1, mp_row, mp_col, num_COM):
    var_dict['mp_COMs1'] = mp_COMs1
    var_dict['mp_row'] = mp_row
    var_dict['mp_col'] = mp_col
    var_dict['num_COM'] = num_COM


def find_delete_Array(i: int):
    num_COM = var_dict['num_COM']
    COMs1 = np.frombuffer(var_dict['mp_COMs1'])
    COMs = COMs1.reshape((num_COM, 2))
    col = np.frombuffer(var_dict['mp_col'])
    row = np.frombuffer(var_dict['mp_row'])
    inds = (col == i).nonzero()[0]
    if inds.size > 1:
        used = row[inds].astype('int')
        r2 = pairwise_distances(COMs[used], COMs[i:i + 1]).squeeze()
        keep = r2.argmin()
        ind_delete = np.delete(inds, keep)
        return ind_delete
    else:
        return np.array([], dtype='int')


def uniqueNeurons1(segs: list, thresh_COM0, useMP=True):
    totalmasks = sparse.vstack([x[0] for x in segs])
    neuronstate = np.hstack([x[1] for x in segs])
    COMs = np.vstack([x[2] for x in segs])
    probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
    maxN = neuronstate.size
    coreneurons = neuronstate.nonzero()[0]
    rows, cols = [], []

    cnt = 0
    keeplist = np.zeros(maxN, dtype='bool')
    for c in coreneurons:
        if neuronstate[c]:
            if useMP: 
                r = np.zeros(COMs.shape[0])
                fastCOMdistance(COMs, COMs[c], r)
            else:
                r = pairwise_distances(COMs, COMs[c:c + 1]).squeeze() 
            neighbors = (r <= thresh_COM0).nonzero()[0]
            neuronstate[neighbors] = False
            cols.append(neighbors)
            rows.append(c * np.ones(neighbors.size))
            keeplist[c] = True
            cnt += 1

    ind_row = np.hstack(rows).astype('int')
    ind_col = np.hstack(cols).astype('int')

    if useMP:    # Use mp.Array
        print('Use mp.Array')
        num_COM = COMs.shape[0]
        mp_COMs = mp.RawArray('d', COMs.ravel())
        mp_row = mp.RawArray('d', ind_row)
        mp_col = mp.RawArray('d', ind_col)
        p2 = mp.Pool(round(mp.cpu_count()), initializer=init_pool, initargs=(mp_COMs, mp_row, mp_col, num_COM))
        ind_delete = p2.map(find_delete_Array, range(maxN), chunksize=16)
        p2.close()
        p2.join()
    else:   ## Use for loop
        print('Use for loop')
        ind_delete = []
        for i in range(maxN):
            inds = (ind_col == i).nonzero()[0]
            if inds.size > 1:
                used = ind_row[inds]
                r2 = pairwise_distances(COMs[used], COMs[i:i + 1]).squeeze()
                keep = r2.argmin()
                inds=np.delete(inds, keep)
                ind_delete.append(inds)

    val = np.ones(ind_row.size)
    ind_delete = np.hstack(ind_delete)
    val[ind_delete] = False

    comb = sparse.csr_matrix((val, (ind_row, ind_col)), (maxN, maxN))
    comb = comb[keeplist]
    uniques = comb.dot(totalmasks)
    times = [probmapID[comb[ii].todense().A.astype('bool').squeeze()] for ii in range(cnt)]

    return uniques, times


def uniqueNeurons1_simp(segs: list, thresh_COM0, useMP=True): #minArea, 
    totalmasks = sparse.vstack([x[0] for x in segs])
    neuronstate = np.hstack([x[1] for x in segs])
    COMs = np.vstack([x[2] for x in segs])
    areas = np.hstack([x[3] for x in segs])
    probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
    # area_select = (areas > minArea).squeeze()
    # totalmasks = totalmasks[area_select]
    # neuronstate = neuronstate[area_select]
    # COMs = COMs[area_select]
    # areas = areas[area_select]
    # probmapID = probmapID[area_select]

    maxN = neuronstate.size
    if maxN>0:
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
                neighbors = np.logical_and(r <= thresh_COM0, unusedneurons).nonzero()[0]
                neuronstate[neighbors] = False
                unusedneurons[neighbors] = False
                cols.append(neighbors)
                rows.append(c * np.ones(neighbors.size))
                keeplist[c] = True
                cnt += 1

        ind_row = np.hstack(rows).astype('int')
        ind_col = np.hstack(cols).astype('int')
        val = np.ones(ind_row.size)

        comb = sparse.csr_matrix((val, (ind_row, ind_col)), (maxN, maxN))
        comb = comb[keeplist]
        uniques = comb.dot(totalmasks)
        times = [probmapID[comb[ii].todense().A.astype('bool').squeeze()] for ii in range(cnt)]
        
    else:
        uniques = Masks_2 = sparse.csr_matrix((0,totalmasks.shape[1]), dtype='bool')
        times = []
    return uniques, times


def uniqueNeurons2_simp(totalmasks:sparse.csr_matrix, neuronstate:np.array, COMs:np.array, areas:np.array, probmapID:np.array, minArea, thresh_COM0, useMP=True):
    # totalmasks = sparse.vstack([x[0] for x in segs])
    # neuronstate = np.hstack([x[1] for x in segs])
    # COMs = np.vstack([x[2] for x in segs])
    # areas = np.hstack([x[3] for x in segs])
    # probmapID = np.hstack([ind * np.ones(x[1].size, dtype='uint32') for (ind, x) in enumerate(segs)])
    # area_select = (areas > minArea).squeeze()
    area_select = (areas > minArea)
    totalmasks = totalmasks[area_select]
    neuronstate = neuronstate[area_select]
    COMs = COMs[area_select]
    areas = areas[area_select]
    probmapID = probmapID[area_select]

    maxN = neuronstate.size
    if maxN>0:
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
                neighbors = np.logical_and(r <= thresh_COM0, unusedneurons).nonzero()[0]
                neuronstate[neighbors] = False
                unusedneurons[neighbors] = False
                cols.append(neighbors)
                rows.append(c * np.ones(neighbors.size))
                keeplist[c] = True
                cnt += 1

        ind_row = np.hstack(rows).astype('int')
        ind_col = np.hstack(cols).astype('int')
        val = np.ones(ind_row.size)

        comb = sparse.csr_matrix((val, (ind_row, ind_col)), (maxN, maxN))
        comb = comb[keeplist]
        uniques = comb.dot(totalmasks)
        times = [probmapID[comb[ii].todense().A.astype('bool').squeeze()] for ii in range(cnt)]
        
    else:
        uniques = Masks_2 = sparse.csc_matrix((0,totalmasks.shape[1]), dtype='bool')
        times = []
    return uniques, times


def group_neurons(uniques: sparse.csr_matrix, thresh_COM, thresh_mask, dims: tuple, times: list, useMP=True):
    N = uniques.shape[0]
    uniques_thresh = sparse.vstack([x >= x.max() * thresh_mask for x in uniques])
    COMs = np.zeros((N, 2))
    for nn in range(N):
        inds = uniques_thresh[nn].nonzero()[1]
        xxs = inds // (dims[1])
        yys = inds % (dims[1])
        COMs[nn] = [xxs.mean(), yys.mean()]

    uniquelist = np.ones(N, dtype='bool')
    keeplist = np.zeros(N, dtype='bool')
    cnt = 0
    rows, cols = [], []
    for i in range(N):
        if uniquelist[i]:
            if useMP: 
                r = np.zeros(COMs.shape[0])
                fastCOMdistance(COMs, COMs[i], r)
            else:
                r = pairwise_distances(COMs, COMs[i:i + 1]).squeeze() 
            # r = pairwise_distances(COMs, COMs[i:i + 1]).squeeze()
            neighbors = (r <= thresh_COM).nonzero()[0]
            uniquelist[neighbors] = False
            cols.append(neighbors)
            rows.append(i * np.ones(neighbors.size, dtype='int32'))
            keeplist[i] = True
            cnt += 1

    ind_row = np.hstack(rows).astype('uint32')
    ind_col = np.hstack(cols).astype('uint32')

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
    uniqueout = comb.dot(uniques)
    uniquetimes = [np.hstack([times[ind] for ind in comb[ii].todense().A.nonzero()[1]]) for ii in range(cnt)]
    return uniqueout, uniquetimes


def piece_neurons_IOU(neuronmasks: sparse.csr_matrix, thresh_mask, thresh_IOU, times: list):
    uniquetimes = times.copy()
    N = neuronmasks.shape[0]
    tempmasks = sparse.vstack([x >= x.max() * thresh_mask for x in neuronmasks]).astype('float')
    area_i = tempmasks.dot(tempmasks.T).todense().A
    area = area_i.diagonal()
    area_i = area_i - np.diag(area)
    a1 = np.expand_dims(area, axis=1).repeat(N, axis=1)
    a2 = a1.T
    area_u = a1 + a2 - area_i
    IOU = np.triu(area_i) / area_u

    (x, y) = (IOU >= thresh_IOU).nonzero()
    belongs = np.arange(N)
    for (xi, yi) in zip(x, y):
        xto = belongs[xi]
        yto = belongs[yi]
        if xto != yto:
            addto = min(xto, yto)
            addfrom = max(xto, yto)
            move = (belongs == addfrom)
            belongs[move] = addto
            uniquetimes[addto] = np.hstack([uniquetimes[addfrom], uniquetimes[addto]])
            uniquetimes[addfrom] = np.array([], dtype='uint32')

    remains = np.unique(belongs)
    comb = sparse.csr_matrix((np.ones(N), (belongs, np.arange(N))), (N, N))
    comb = comb[remains]
    neuronmasks = comb.dot(neuronmasks)
    uniquetimes = [uniquetimes[k] for k in remains]

    return neuronmasks, uniquetimes


def piece_neurons_consume(neuronmasks: sparse.csr_matrix, avgA, thresh_mask, thresh_consume, times: list):
    uniquetimes = times.copy()
    N = neuronmasks.shape[0]
    tempmasks = sparse.vstack([x >= x.max() * thresh_mask for x in neuronmasks]).astype('float')
    area_i = tempmasks.dot(tempmasks.T).todense().A
    area = area_i.diagonal()
    area_i = area_i - np.diag(area)
    a2 = np.expand_dims(area, axis=0).repeat(N, axis=0)
    consume = area_i / a2

    belongs = np.arange(N)
    throw = 2 * N + 1
    (x, y) = (consume >= thresh_consume).nonzero()
    for (xi, yi) in zip(x, y):
        area_x = area[xi]
        area_y = area[yi]
        if max(area_x, area_y) > avgA:
            if area_y > area_x:
                belongs[yi] = throw
                uniquetimes[yi] = np.array([], dtype='uint32')
            else:
                belongs[xi] = throw
                uniquetimes[xi] = np.array([], dtype='uint32')
        else:
            xto = belongs[xi]
            yto = belongs[yi]
            if xto != yto:
                addto = min(xto, yto)
                addfrom = max(xto, yto)
                move = (belongs == addfrom)
                belongs[move] = addto
                uniquetimes[addto] = np.hstack([uniquetimes[addfrom], uniquetimes[addto]])
                uniquetimes[addfrom] = np.array([], dtype='uint32')

    keep = (belongs != throw).nonzero()[0]
    comb = sparse.csr_matrix((np.ones(keep.size), (belongs[keep], np.arange(N)[keep])), (N, N))
    remains = np.setdiff1d(belongs, throw)
    comb = comb[remains]
    neuronmasks = comb.dot(neuronmasks)
    uniquetimes = [uniquetimes[k] for k in remains]

    return neuronmasks, uniquetimes

