# %%
import sys
import numpy as np
import h5py

sys.path.insert(1, '../..') # the path containing "suns" folder
from suns.PreProcessing.par1 import fastconv

# %%
if __name__ == '__main__':
    filename_TF_template = '01_spike_tempolate.h5' # File name storing the temporal filter kernel
    h5f = h5py.File(filename_TF_template,'r')
    Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    Poisson_filt = Poisson_filt/Poisson_filt.sum()

    T = 1500
    L = 30
    for _ in range(9):
        f = np.random.random(L).astype('float32')
        a = np.random.random((T,512,512)).astype('float32')
        b = np.zeros((T-L+1,512,512), dtype='float32')
        print('start')
        fastconv(a,b,f)
        print('finish')
        del f, a, b
