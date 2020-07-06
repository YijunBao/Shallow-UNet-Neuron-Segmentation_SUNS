import os
import cv2
import numpy as np
import time
import multiprocessing as mp
import pyfftw
from numba import njit, prange


dir_wisdom = 'wisdom\\'
if not os.path.exists(dir_wisdom):
    os.makedirs(dir_wisdom) 
Dimens = [(224,224),(216,152), (248,248),(120,88)]
Nframes = [90000, 41000, 116043, 3000]

ind_video = 2
start = time.time()
rows, cols = Dimens[ind_video] # 487, 487, #img1.shape
x = cv2.getOptimalDFTSize(rows)
y = cv2.getOptimalDFTSize(cols)
nn = Nframes[ind_video] # 900
#m, n = 512, 512
# nn = 20000

start1 = time.time()
bb = pyfftw.zeros_aligned((nn, x, y), dtype='float32', n=8)
bf = pyfftw.zeros_aligned((nn, x, y//2+1), dtype='complex64', n=8)
fft_object_b = pyfftw.FFTW(bb, bf,axes=(-2,-1),flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=mp.cpu_count())
fft_object_c = pyfftw.FFTW(bf, bb,axes=(-2,-1),flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD',threads=mp.cpu_count())
end1 = time.time()

bb = pyfftw.export_wisdom()
print(bb)
Length_data=str((nn, x, y))
file = open(dir_wisdom+Length_data+"x1.txt", "wb")
file.write(bb[0])
file.close
file = open(dir_wisdom+Length_data+"x2.txt", "wb")
file.write(bb[1])
file.close
file = open(dir_wisdom+Length_data+"x3.txt", "wb")
file.write(bb[2])
file.close()
print(end1-start1, ' s')
