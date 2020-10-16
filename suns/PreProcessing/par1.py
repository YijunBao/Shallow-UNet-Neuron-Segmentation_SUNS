import numpy as np
import numba
from numba import jit, prange, f4, u1, c8
import math


@jit("void(f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastlog(f):
    '''Step 1 of FFT-based spatial filtering: computing the log of the input video.

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video

    Outputs:
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the output video, log(1+f)
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                f[i,j,k] = math.log1p(f[i,j,k])


@jit("void(c8[:,:,:],f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmask(f, mask):
    '''Step 3 of FFT-based spatial filtering: 
        multiplying the input video with a 2D mask, which is a 2D Gaussian function.

    Inputs: 
        f(numpy.ndarray of complex64, shape = (T,Lx,Ly)): the input video
        mask(numpy.ndarray of float32, shape = (Lx,Ly)): 2D array of spatial filter mask

    Outputs:
        f(numpy.ndarray of complex64, shape = (T,Lx,Ly)): element-wise multiplication of f*mask
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                f[i,j,k] = f[i,j,k] * mask[j,k]


@jit("void(f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastexp(f):
    '''Step 5 of FFT-based spatial filtering: computing the exp of the input video.

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video

    Outputs:
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the output video, exp(f)
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                f[i,j,k] = math.exp(f[i,j,k])


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


@jit("void(f4[:,:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True)
def fastmedian(A,b):
    '''Calculate the medians of the temporal traces of each pixel

    Inputs: 
        A(numpy.ndarray of float32, shape = (Lx,Ly,T)): the input video

    Outputs:
        b(numpy.ndarray of float32, shape = (Lx,Ly,1)): the medians of each pixel
    '''
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            b[i,j] = np.median(A[i,j,:])


@jit("void(f4[:,:,:],f4[:],f4[:,:,:])",nopython=True,parallel=True,cache=True)
def fastquant(A,q,b):
    '''Calculate the quantiles of the temporal traces of each pixel

    Inputs: 
        A(numpy.ndarray of float32, shape = (Lx,Ly,T)): the input video
        q(numpy.ndarray of float32, shape = (n,)): 1D array of quantile ratios

    Outputs:
        b(numpy.ndarray of float32, shape = (Lx,Ly,n)): the quantiles of each pixel
    '''
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            b[i,j] = np.quantile(A[i,j,:],q)


@jit("void(f4[:,:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastnormf(f, meds):
    '''Normalize the input video pixel-by-pixel into SNR video.
        f(t,x,y) = (f(t,x,y) - median(x,y))/median_based_std(x,y)

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video
        meds(numpy.ndarray of float32, shape = (2,Lx,Ly)): the median and median-based std
            meds[0,:,:] is the median
            meds[1,:,:] is the median-based std

    Outputs:
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): becomes the SNR video
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                f[i,j,k] = (f[i,j,k] - meds[0,j,k])*meds[1,j,k]


@jit("void(f4[:,:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.float32})
def fastnormback(f, mu):
    '''Normalize the input video by the mean of the median of every pixel.
        f(t,x,y) = f(t,x,y)/mu.
        This function is used when SNR normalization is not used.

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video
        mu(float32): the mean of pixel-wise median of the video

    Outputs:
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the output video, f/mu
    '''
    mu_1 = 1/mu
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                f[i,j,k] = f[i,j,k]*mu_1


@jit("void(f4[:,:,:], f4[:,:], u4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmediansubtract(f, temp, dec):
    '''Subtract every frame with its median, to remove large-scale fluctuation.

    Inputs: 
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the input video
        dec(int): the median is calculated every "dec" pixels to save time

    Outputs:
        f(numpy.ndarray of float32, shape = (T,Lx,Ly)): the output video, after median subtraction
    '''
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            temp[i,j] = np.median(f[i, j, ::dec])
        f[i] -= np.median(temp[i, ::dec])

