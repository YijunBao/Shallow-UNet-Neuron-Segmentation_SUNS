
# %%
import time
import numpy as np
from scipy.io import loadmat
import h5py
import fissa
import matplotlib.pyplot as plt

# %%
if __name__ == '__main__':
    Exp_ID = '539670003'
    file_video = 'C:\\Matlab Files\\Filter\\video_SNR.dat'
    Lx = 487
    Ly = 487
    nframes = 230
    fid = open(file_video, 'rb')
    images = np.fromfile(fid, np.float32, Lx * Ly * nframes)
    images = images.reshape((nframes, Ly, Lx))
    images = images - images.min()
    # images = (images*128).astype('int16')

    file_mask = r'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\FinalMasks_FPremoved_' + Exp_ID + '.mat'
    # GTMasks = loadmat(file_mask)
    mat=h5py.File(file_mask,'r')
    rois = np.array(mat['FinalMasks']).astype('bool')[:44]
    folder = 'FISSA'
    
# %% experiment definition
if __name__ == '__main__':
    start = time.time()
    experiment = fissa.Experiment([images], [rois.tolist()], folder)
    init = time.time()
    experiment.separation_prep(redo=True)
    prep = time.time()
    experiment.separate(redo_prep=False, redo_sep=True)
    finish = time.time()
    experiment.save_to_matlab()
    print('Initilization time:', init-start,'s')
    print('Preparation time:', prep-init,'s')
    print('Separation time:', finish-prep,'s')
    print('Total FISSA time:', finish-start,'s')

# %%
    thred_ratio = 5
    ncells = rois.shape[0]
    raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
    sep_traces = np.vstack([experiment.sep[x][0][0] for x in range(ncells)])
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_raw = med_unmix[0]
    sigma_unmix = (med_raw[1]-med_raw[1])#/(math.sqrt(2)*special.erfinv(0.5))
    thred = np.expand_dims(mu_raw + thred_ratio*sigma_unmix, axis=1)
    active = unmixed_traces > thred
    

# %%
    preparation = np.load(folder+'\\preparation.npy', allow_pickle=True)
    separated = np.load(folder+'\\separated.npy', allow_pickle=True)
    raw_traces = np.vstack([preparation[1][x][0][0] for x in range(ncells)])
    sep_traces = np.vstack([separated[2,x,0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([separated[3,x,0][0] for x in range(ncells)])
    # print('+1s')

# %%
