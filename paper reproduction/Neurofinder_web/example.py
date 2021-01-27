# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#

import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
from glob import glob

# load the images
files = sorted(glob('images/*.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

# load the regions (training data only)
with open('regions/regions.json') as f:
    regions = json.load(f)

def tomask(coords):
    mask = zeros(dims)
    mask[zip(*coords)] = 1
    return mask

masks = array([tomask(s['coordinates']) for s in regions])

# show the outputs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imgs.sum(axis=0), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(masks.sum(axis=0), cmap='gray')
plt.show()