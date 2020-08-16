import numba
from numba import jit, prange, f8, f4, u1
import math


@jit("void(f8[:,:],f8[:],f8[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastCOMdistance(a,b,d):
    '''Calculate the COM distances between a point "b" and a series of points "a"

    Inputs: 
        a(numpy.ndarray of float64, shape = (L,2)): The COMs of a a series point
        b(numpy.ndarray of float64, shape = (2,)): The COM of a point

    Outputs:
        d(numpy.ndarray of float64, shape = (L,)): the COM distances between "b" and each point in "a"
    '''
    x = b[0]
    y = b[1]
    for i in prange(a.shape[0]):            
        d[i] = math.sqrt((a[i,0]-x)**2+(a[i,1]-y)**2)


@jit("void(f4[:,:,:],u1[:,:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastthreshold(f, g, th):
    '''Binarize the input video using a threshold.
        When a value is larger than the threshold, it is set as 255;
        When a value is smaller than or equal to the threshold, it is set as 0.

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video
        th(float32): the mean of pixel-wise median of the video

    Outputs:
        g(numpy.ndarray of uint8, shape = (T,Lx,Ly)): the thresholded video
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                if f[i,j,k] > th:
                    g[i,j,k] = 255
                else:
                    g[i,j,k] = 0
