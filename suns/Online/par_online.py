import numpy as np
import numba
from numba import jit, prange, f4, c8, u1
import math


@jit("void(f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastlog_2(f):
    '''Step 1 of FFT-based spatial filtering: computing the log of the input video.

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video

    Outputs:
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the output video, log(1+f)
    '''
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = math.log1p(f[j,k])


@jit("void(c8[:,:],f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmask_2(f, mask):
    '''Step 3 of FFT-based spatial filtering: 
        multiplying the input video with a 2D mask, which is a 2D Gaussian function.

    Inputs: 
        f(numpy.ndarray of complex64, shape = (Lx,Ly)): the input video
        mask(numpy.ndarray of float32, shape = (Lx,Ly)): 2D array of spatial filter mask

    Outputs:
        f(numpy.ndarray of complex64, shape = (Lx,Ly)): element-wise multiplication of f*mask
    '''
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = f[j,k] * mask[j,k]


@jit("void(f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastexp_2(f):
    '''Step 5 of FFT-based spatial filtering: computing the exp of the input video.

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video

    Outputs:
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the output video, exp(f)
    '''
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = math.exp(f[j,k])


@jit("void(f4[:,:,:],f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.float32})
def fastconv_2(a,b,f):
    '''Online temporal filtering. Convolve "a" with flipped viersion of "b"

    Inputs: 
        a(numpy.ndarray of float32, shape = (nt,Lx,Ly)): the input video
        f(numpy.ndarray of float32, shape = (nt,)): 1D array of temporal filter kernel

    Outputs:
        b(numpy.ndarray of float32, shape = (Lx,Ly)): the output video, convolution of a and f
    '''
    for j in prange(a.shape[1]):
        for k in prange(a.shape[2]):
            temp = 0
            for l in prange(len(f)):
                temp += a[l,j,k]*f[l]
            b[j,k]=temp


@jit("void(f4[:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastnormf_2(f, meds):
    '''Normalize the input video pixel-by-pixel into SNR video.
        f(x,y) = (f(x,y) - median(x,y))/median_based_std(x,y)

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video
        meds(numpy.ndarray of float32, shape = (2,Lx,Ly)): the median and median-based std
            meds[0,:,:] is the median
            meds[1,:,:] is the median-based std

    Outputs:
        f(numpy.ndarray of float32, shape = (Lx,Ly)): becomes the SNR video
    '''
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
                f[j,k] = (f[j,k] - meds[0,j,k])*meds[1,j,k]


@jit("void(f4[:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastnormback_2(f, mu):
    '''Normalize the input video into SNR video.
        f(x,y) = f(x,y)/mu.
        This function is used when SNR normalization is not used.

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video
        mu(float32): the mean of pixel-wise median of the video

    Outputs:
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the output video, f/mu
    '''
    mu_1 = 1/mu
    for j in prange(f.shape[0]):
        for k in prange(f.shape[0]):
            f[j,k] = f[j,k]*mu_1


@jit("void(f4[:,:],u1[:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastthreshold_2(f, g, th):
    '''Binarize the input video using a threshold.
        When a value is larger than the threshold, it is set as 255;
        When a value is smaller than or equal to the threshold, it is set as 0.

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video
        th(float32): the mean of pixel-wise median of the video

    Outputs:
        g(numpy.ndarray of uint8, shape = (Lx,Ly)): the thresholded video
    '''
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            if f[j,k] > th:
                g[j,k] = 255
            else:
                g[j,k] = 0


@jit("void(f4[:,:], f4[:], u4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmediansubtract_2(f, temp, dec):
    '''Subtract a frame with its median, to remove large-scale fluctuation.

    Inputs: 
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the input video
        dec(int): the median is calculated every "dec" pixels to save time

    Outputs:
        f(numpy.ndarray of float32, shape = (Lx,Ly)): the output video, after median subtraction
    '''
    for i in prange(f.shape[0]):
        temp[i] = np.median(f[i, ::dec])
    f -= np.median(temp)

