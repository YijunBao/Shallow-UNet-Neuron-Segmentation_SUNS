import os
import cv2
import numpy as np
import time
import multiprocessing as mp
import pyfftw


'''Learn wisdom of a 2D array, used for fast FFT planning'''
Dimens = (120,88) # The lateral dimension of the target 2D array
dir_wisdom = 'wisdom\\'
if not os.path.exists(dir_wisdom):
    os.makedirs(dir_wisdom) 

start = time.time()
rows, cols = Dimens
x = cv2.getOptimalDFTSize(rows)
y = cv2.getOptimalDFTSize(cols)

start1 = time.time()
bb = pyfftw.zeros_aligned((x, y), dtype='float32', n=8)
bf = pyfftw.zeros_aligned((x, y//2+1), dtype='complex64', n=8)
fft_object_b = pyfftw.FFTW(bb, bf,axes=(-2,-1),flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=mp.cpu_count())
fft_object_c = pyfftw.FFTW(bf, bb,axes=(-2,-1),flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD',threads=mp.cpu_count())
end1 = time.time()

bb = pyfftw.export_wisdom()
print(bb)
Length_data=str((x, y))
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
