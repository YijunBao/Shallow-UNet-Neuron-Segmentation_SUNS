# %%
import sys
import numpy as np
import h5py
from cpuinfo import get_cpu_info
import sys

sys.path.insert(1, '../..') # the path containing "suns" folder
from suns.PreProcessing.par1 import fastconv

# %%
if __name__ == '__main__':
    print(get_cpu_info()['brand_raw'])
    print('win' in sys.platform.lower() and 'AMD' in get_cpu_info()['brand_raw'].upper())
    T = 1500
    L = 30
    X = 256
    for _ in range(9):
        f = np.random.random(L).astype('float32')
        a = np.random.random((T,X,X)).astype('float32')
        b = np.zeros((T-L+1,X,X), dtype='float32')
        print('start')
        fastconv(a,b,f)
        print('finish')
        del f, a, b
