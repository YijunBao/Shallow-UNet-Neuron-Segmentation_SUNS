import numpy as np
import numba
from numba import jit, prange
import math
# numba.config.NUMBA_NUM_THREADS=12

# @jit("void(f4[:,:,:],f4[:,:,:],f4,u1[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'a': numba.int32})
# def fastnorm(f, meds,baseline, output):
#     for i in range(output.shape[0]):
#         for j in prange(output.shape[1]):
#             for k in prange(output.shape[2]):
#                 a = np.round((f[i,j,k] - meds[int(0),j,k])*meds[1,j,k] + baseline)
#                 if a > 255:
#                     a = 255
#                 elif a < 0:
#                     a = 0
#                 output[i,j,k] = a

@jit("void(f4[:,:],f4[:,:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastnormf(f, meds):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
                f[j,k] = (f[j,k] - meds[0,j,k])*meds[1,j,k]

@jit("void(f4[:,:],f4[:],f4[:,:])",nopython=True,parallel=True,cache=True)
def fastquant(A,q,b):
    for j in prange(A.shape[0]):
        b[j] = np.quantile(A[j,:],q)

# @jit("void(c8[:,:,:],f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
# def fastmask1(f, mask):
#     for i in prange(f.shape[0]):
#         f[i] = f[i] * mask

@jit("void(c8[:,:],f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastmask(f, mask):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = f[j,k] * mask[j,k]

@jit("void(f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastlog(f):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = math.log1p(f[j,k])

@jit("void(f4[:,:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fastexp(f):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            f[j,k] = math.exp(f[j,k])
            # f[i,j,k] = math.expm1(f[i,j,k])

# @jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
# def fastconv1(a,b,f):
#     for i in range(a.shape[0]-len(f)+1):
#         # b[i] = a[i]*f[0]+a[i+1]*f[1]+a[i+2]*f[2]+ a[i+3]*f[3]+ a[i+4]*f[4]+a[i+5]*f[5]+ a[i+6]*f[6]+a[i+7]*f[7]+a[i+8]*f[8]+a[i+9]*f[9]+a[i+10]*f[10]+a[i+11]*f[11]+a[i+12]*f[12]+a[i+13]*f[13]+a[i+14]*f[14]+a[i+15]*f[15]
#         b[i] = a[i]*f[0]+a[i+1]*f[1]+a[i+2]*f[2]+ a[i+3]*f[3]+ a[i+4]*f[4]+a[i+5]*f[5]

# @jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.uint32})
# def fastconv(a,b,f):
#     for i in range(a.shape[0]-len(f)+1):
#         for j in prange(a.shape[1]):
#             for k in prange(a.shape[2]):                
#                 b[i,j,k] = (a[i:i+len(f),j,k]*f).sum()

@jit("void(f4[:,:,:],f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True,locals={'temp': numba.float32})
def fastconv(a,b,f):
    for j in prange(a.shape[1]):
        for k in prange(a.shape[2]):
            temp = 0
            for l in prange(len(f)):
                temp += a[l,j,k]*f[l]
            b[j,k]=temp

# @jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=False,cache=True,fastmath=True,locals={'temp': numba.float32})
# def slowconv(a,b,f):
#     for i in range(a.shape[0]-len(f)+1):
#         for j in range(a.shape[1]):
#             for k in range(a.shape[2]):
#                 temp = 0
#                 for l in range(f.size):
#                     temp += a[i+l,j,k]*f[l]
#                 b[i,j,k]=temp

# @jit("void(f4[:,:,:],f4[:,:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
# def fastconv4(a,b,f):
#     for i in range(a.shape[0]-len(f)+1):
#         for j in prange(a.shape[1]):
#             for k in prange(a.shape[2]):                
#                 b[i,j,k] = a[i,j,k]*f[0]+a[i+1,j,k]*f[1]+a[i+2,j,k]*f[2]+ a[i+3,j,k]*f[3]+ a[i+4,j,k]*f[4]+a[i+5,j,k]*f[5]

@jit("void(f4[:,:],f4,f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastnormback(f, mu, sigma):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[0]):
            f[j,k] = (f[j,k]-mu)/sigma

@jit("void(f4[:,:],u1[:,:],f4)",nopython=True,parallel=True,cache=True,fastmath=True)
def fastthreshold(f, g, th):
    for j in prange(f.shape[0]):
        for k in prange(f.shape[1]):
            if f[j,k] > th:
                g[j,k] = 255
            else:
                g[j,k] = 0