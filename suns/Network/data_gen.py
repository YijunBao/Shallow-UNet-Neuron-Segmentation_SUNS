import random
import numpy as np


def data_gen(train_img, train_mask, batch_size, flips=False, rotate=False): 
    '''Generator of training images and masks from an array dataset that can be non-square.
        Used for CNN training.

    Inputs: 
        train_img(3D numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR images
        train_mask(3D numpy.ndarray of uint8, shape = (T,Lx,Ly)): the temporal masks
        batch_size(int): batch size for training
        flips(bool, default to False): Indicator of whether random filpping is used.
        rotate(bool, default to False): Indicator of whether random rotation is used.

    Outputs:
        train_img(4D numpy.ndarray of float32, shape = (batch_size,L1,L2,1)): the SNR images
        train_mask(4D numpy.ndarray of uint8, shape = (batch_size,L1,L2,1)): the temporal masks
            The lateral sizes can be (L1,L2) = (Lx,Ly) or (L1,L2) = (Ly,Lx)
    '''
    print('flips = {}, rotate = {}'.format(flips, rotate))
    c = 0
    n_images, rows, cols = train_img.shape
    list_n = list(range(0,n_images))
    random.seed()
    random.shuffle(list_n) # randomize the order of images
    rotate_each = rotate and (rows==cols) 
    rotate_all = rotate and (rows!=cols) 

    while (True):
        img = np.zeros((batch_size, rows, cols, 1), dtype=train_img.dtype)
        mask = np.zeros((batch_size, rows, cols, 1), dtype=train_mask.dtype)
        for i in range(c, c + batch_size):
            current_n = list_n[i]
            test_img = train_img[current_n, :, :]
            test_mask = train_mask[current_n, :, :]
            p_hflip = 0
            p_vflip = 0
            p_rotate = 0
            if flips:
                p_hflip = random.random()
                p_vflip = random.random()

            if rotate_each: # When rows==cols, each frame can be rotated seperately
                p_rotate = random.random()

            if p_hflip > 0.5: # flip horizontally with probablity of 0.5
                test_img = np.fliplr(test_img)
                test_mask = np.fliplr(test_mask)
            if p_vflip > 0.5: # flip vertically with probablity of 0.5
                test_img = np.flipud(test_img)
                test_mask = np.flipud(test_mask)
            if p_rotate > 0.5: # rotate by 90 degrees with probablity of 0.5
                test_img = np.rot90(test_img)
                test_mask = np.rot90(test_mask)

            img[i - c, :, :, 0] = test_img
            mask[i - c, :, :, 0] = test_mask

        if rotate_all: # When rows!=cols, the frames must be rotated uniformly
            p_rotate = random.random()
            if p_rotate > 0.5: # rotate by 90 degrees with probablity of 0.5
                img = np.rot90(img, axes=(1,2))
                mask = np.rot90(mask, axes=(1,2))

        c += batch_size
        if (c + batch_size > n_images): 
            # When all the images are used up, start again from beginning with new shuffle
            c = 0
            random.shuffle(list_n)
        yield img, mask


def data_gen_list(list_train_img, list_train_mask, batch_size, flips=False, rotate=False): 
    '''Generator of training images and masks from a list of datasets with different sizes.
        Used for CNN training.

    Inputs: 
        list_train_img(list of 3D numpy.ndarray of float32, shape = (T,Lx,Ly)): list of SNR images
        list_train_mask(list of 3D numpy.ndarray of uint8, shape = (T,Lx,Ly)): list of temporal masks
            Each array in the list must have the same T, but can have different Lx and Ly.
        batch_size(int): batch size for training
        flips(bool, default to False): Indicator of whether random filpping is used.
        rotate(bool, default to False): Indicator of whether random rotation is used.

    Outputs:
        train_img(4D numpy.ndarray of float32, shape = (batch_size,L1,L2,1)): the SNR images
        train_mask(4D numpy.ndarray of uint8, shape = (batch_size,L1,L2,1)): the temporal masks
            The lateral sizes can be (L1,L2) = (Lx,Ly) or (L1,L2) = (Ly,Lx)
    '''
    print('flips = {}, rotate = {}'.format(flips, rotate))
    num_videos = len(list_train_img) # number of arrays
    list_c = [0]*num_videos # number of used images in each array
    n_images = list_train_img[0].shape[0] # number of images for each array
    list_n = list(range(0,n_images))
    list_shuffle = []
    list_remain = list(range(num_videos))
    random.seed()
    for n in range(num_videos):
        random.shuffle(list_n) # randomize the order of images in each array
        list_shuffle.append(list_n.copy())

    while (True):
        # randomly choose an array, and use all the images in this array
        n = random.choice(list_remain) 
        train_img = list_train_img[n]
        train_mask = list_train_mask[n]
        _, rows, cols = train_img.shape
        img = np.zeros((batch_size, rows, cols, 1), dtype='float32')
        mask = np.zeros((batch_size, rows, cols, 1), dtype='uint8')
        rotate_each = rotate and (rows==cols) 
        rotate_all = rotate and (rows!=cols) 
        c = list_c[n]
        for i in range(c, c + batch_size):
            current_n = list_shuffle[n][i]
            test_img = train_img[current_n, :, :]
            test_mask = train_mask[current_n, :, :]
            p_hflip = 0
            p_vflip = 0
            p_rotate = 0
            if flips:
                p_hflip = random.random()
                p_vflip = random.random()

            if rotate_each: # When rows==cols, each frame can be rotated seperately
                p_rotate = random.random()

            if p_hflip > 0.5: # flip horizontally with probablity of 0.5
                test_img = np.fliplr(test_img)
                test_mask = np.fliplr(test_mask)
            if p_vflip > 0.5: # flip vertically with probablity of 0.5
                test_img = np.flipud(test_img)
                test_mask = np.flipud(test_mask)
            if p_rotate > 0.5: # rotate by 90 degrees with probablity of 0.5
                test_img = np.rot90(test_img)
                test_mask = np.rot90(test_mask)

            img[i - c, :, :, 0] = test_img
            mask[i - c, :, :, 0] = test_mask

        if rotate_all: # When rows!=cols, the frames must be rotated uniformly
            p_rotate = random.random()
            if p_rotate > 0.5: # rotate by 90 degrees with probablity of 0.5
                img = np.rot90(img, axes=(1,2))
                mask = np.rot90(mask, axes=(1,2))

        list_c[n] += batch_size
        if (list_c[n] + batch_size > n_images): # all the images are used up
            list_remain.remove(n)

        if not list_remain: 
            # When all the arrays are used up, start again from beginning with new shuffle
            list_remain = list(range(num_videos))
            for n in range(num_videos):
                random.shuffle(list_shuffle[n])
            list_c = [0]*num_videos
        yield img, mask


