# %%
import numpy as np
import numba
from numba import jit, prange, f4


@jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.float32})
def fastconv(a,b,f):
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
    T = 500
    L = 30
    X = 512
    for _ in range(9):
        f = np.random.random(L).astype('float32')
        a = np.random.random((T,X,X)).astype('float32')
        b = np.zeros((T-L+1,X,X), dtype='float32')
        print('start')
        fastconv(a,b,f)
        print('finish')
        del f, a, b
