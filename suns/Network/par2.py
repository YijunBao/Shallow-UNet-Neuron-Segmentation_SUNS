import numba
from numba import jit, prange, f4, u1
import math


@jit("void(f4[:,:,:],u1[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastuint(pfloat, puint):
    '''Convert an array from float32 to uint8.

    Inputs: 
        pfloat(numpy.ndarray of float32, shape = (T,Lx,Ly)): the video to be converted

    Outputs:
        puint(numpy.ndarray of uint8, shape = (T,Lx,Ly)): the converted video
    '''
    for i in prange(pfloat.shape[0]):
        for j in prange(pfloat.shape[1]):
            for k in prange(pfloat.shape[2]):
                # puint[i,j,k] = pfloat[i,j,k]*256-0.5
                puint[i,j,k] = pfloat[i,j,k]*256 # for published version


@jit("void(f4[:,:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastcopy(p1, p2):
    '''Copy an array to another array.

    Inputs: 
        p1(numpy.ndarray of float32, shape = (T,Lx,Ly)): the video to be copied

    Outputs:
        p2(numpy.ndarray of float32, shape = (T,Lx,Ly)): the copied video
    '''
    for i in prange(p1.shape[0]):
        for j in prange(p1.shape[1]):
            for k in prange(p1.shape[2]):
                p2[i,j,k] = p1[i,j,k]