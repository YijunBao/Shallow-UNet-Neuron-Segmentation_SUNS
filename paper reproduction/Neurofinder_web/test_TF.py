# %%
import sys
import numpy as np
import h5py
# from cpuinfo import get_cpu_info
import sys

# sys.path.insert(1, '../..') # the path containing "suns" folder
# from suns.PreProcessing.par1 import fastconv
import numpy as np
import numba
from numba import jit, prange, f4


@jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.float32})
def fastconv(a,b,f):
    '''Temporal filtering. Convolve "a" with flipped viersion of "b"

    Inputs: 
        a(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video
        f(numpy.ndarray of float32, shape = (nt,)): 1D array of temporal filter kernel

    Outputs:
        b(numpy.ndarray of float32, shape = (T-nt+1,Lx,Ly)): the output video, convolution of a and f
    '''
    lf = len(f)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                temp = 0
                for l in prange(lf):
                    temp += a[i+l,j,k]*f[l]
                b[i,j,k]=temp


# %%
if __name__ == '__main__':
    # print(get_cpu_info()['brand_raw'])
    # print('win' in sys.platform.lower() and 'AMD' in get_cpu_info()['brand_raw'].upper())
    T = 1500
    L = 30
    X = 256
    for _ in range(9):
        f = np.random.random(L).astype('float32')
        a = np.random.random((T,X,X)).astype('float32')
        b = np.zeros((T-L+1,X,X), dtype='float32')
        print('start')
        fastconv(a,b,f)
        print('finish')
        del f, a, b
