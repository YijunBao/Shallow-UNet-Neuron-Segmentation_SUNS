import numpy as np
import numba
from numba import jit, prange
import math

@jit("void(u1[:,:,:],u1[:,:,:],u4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmovmean(a,b,lf):
    before = lf//2
    after = lf-before
    (L0, L1, L2) = a.shape
    for i in prange(0, before):
        for j in prange(L1):
            for k in prange(L2):                
                b[i,j,k] = a[0:i+after, j,k].mean()
    for i in prange(before, L0-after):
        for j in prange(L1):
            for k in prange(L2):                
                b[i,j,k] = a[i-before:i+after, j,k].mean()
    for i in prange(L0-after, L0):
        for j in prange(L1):
            for k in prange(L2):                
                b[i,j,k] = a[i-before:L0, j,k].mean()


# @jit("void(f4[:,:],f4[:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
# def fastCOMdistance2(a,b,d):
#     for i in prange(a.shape[0]):            
#         d[i] = math.sqrt(((a[i]-b)**2).sum())

@jit("void(f8[:,:],f8[:],f8[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastCOMdistance(a,b,d):
    x = b[0]
    y = b[1]
    for i in prange(a.shape[0]):            
        d[i] = math.sqrt((a[i,0]-x)**2+(a[i,1]-y)**2)


@jit("void(f4[:,:,:],u1[:,:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastthreshold(f, g, th):
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                if f[i,j,k] > th:
                    g[i,j,k] = 255
                else:
                    g[i,j,k] = 0
