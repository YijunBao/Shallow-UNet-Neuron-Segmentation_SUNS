import numpy as np
import numba
from numba import jit, prange
import math


@jit("void(f4[:,:,:],u1[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastuint(pfloat, puint):
    for i in prange(pfloat.shape[0]):
        for j in prange(pfloat.shape[1]):
            for k in prange(pfloat.shape[2]):
                puint[i,j,k] = pfloat[i,j,k]*256-0.5
                # puint[i,j,k] = pfloat[i,j,k]*256 # for published version


@jit("void(f4[:,:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastcopy(p1, p2):
    for i in prange(p1.shape[0]):
        for j in prange(p1.shape[1]):
            for k in prange(p1.shape[2]):
                p2[i,j,k] = p1[i,j,k]