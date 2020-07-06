# %%
import random
import numpy as np


# %% Generator of training images from an array dataset that can be non-square 
def data_gen(train_img, train_mask, batch_size, flips=False, rotate=False): 
    print('flips = {}, rotate = {}'.format(flips, rotate))
    c = 0
    n_list, rows, cols = train_img.shape
    n = list(range(0,n_list))
    random.seed()
    random.shuffle(n)
    rotate1 = rotate and (rows==cols)

    while (True):
        img = np.zeros((batch_size, rows, cols, 1), dtype='float32')
        mask = np.zeros((batch_size, rows, cols, 1), dtype='uint8')
        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            current_n = n[i]
            test_img = train_img[current_n, :, :]
            test_mask = train_mask[current_n, :, :]
            p_hflip = 0
            p_vflip = 0
            p_rotate = 0
            if flips:
                p_hflip = random.random()
                p_vflip = random.random()

            if rotate1:
                p_rotate = random.random()

            if p_hflip > 0.5:
                test_img = np.fliplr(test_img)
                test_mask = np.fliplr(test_mask)
            if p_vflip > 0.5:
                test_img = np.flipud(test_img)
                test_mask = np.flipud(test_mask)
            if p_rotate > 0.5:
                test_img = np.rot90(test_img)
                test_mask = np.rot90(test_mask)

            img[i - c, :, :, 0] = test_img
            mask[i - c, :, :, 0] = test_mask

        if rotate and (rows!=cols):
            p_rotate = random.random()
            if p_rotate > 0.5:
                img = np.rot90(img, axes=(1,2))
                mask = np.rot90(mask, axes=(1,2))

        c += batch_size
        if (c + batch_size > n_list):
            c = 0
            random.shuffle(n)
            # print "randomizing again"
        yield img, mask


# %% Generator of training images from a list of datasets with different sizes
def data_gen_list(list_train_img, list_train_mask, batch_size, flips=False, rotate=False): 
    print('flips = {}, rotate = {}'.format(flips, rotate))
    num_videos = len(list_train_img)
    list_c = [0]*num_videos
    n_list = list_train_img[0].shape[0]
    list_n = list(range(0,n_list))
    list_shuffle = []
    list_remain = list(range(num_videos))
    random.seed()
    for n in range(num_videos):
        random.shuffle(list_n)
        list_shuffle.append(list_n.copy())

    while (True):
        # n = random.randint(0,num_videos-1)
        n = random.choice(list_remain)
        train_img = list_train_img[n]
        train_mask = list_train_mask[n]
        _, rows, cols = train_img.shape
        img = np.zeros((batch_size, rows, cols, 1), dtype='float32')
        mask = np.zeros((batch_size, rows, cols, 1), dtype='uint8')
        rotate1 = rotate and (rows==cols)
        c = list_c[n]
        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            current_n = list_shuffle[n][i]
            test_img = train_img[current_n, :, :]
            test_mask = train_mask[current_n, :, :]
            p_hflip = 0
            p_vflip = 0
            p_rotate = 0
            if flips:
                p_hflip = random.random()
                p_vflip = random.random()

            if rotate1:
                p_rotate = random.random()

            if p_hflip > 0.5:
                test_img = np.fliplr(test_img)
                test_mask = np.fliplr(test_mask)
            if p_vflip > 0.5:
                test_img = np.flipud(test_img)
                test_mask = np.flipud(test_mask)
            if p_rotate > 0.5:
                test_img = np.rot90(test_img)
                test_mask = np.rot90(test_mask)

            img[i - c, :, :, 0] = test_img
            mask[i - c, :, :, 0] = test_mask

        if rotate and (rows!=cols):
            p_rotate = random.random()
            if p_rotate > 0.5:
                img = np.rot90(img, axes=(1,2))
                mask = np.rot90(mask, axes=(1,2))

        list_c[n] += batch_size
        if (list_c[n] + batch_size > n_list):
            list_remain.remove(n)

        if not list_remain:
            list_remain = list(range(num_videos))
            for n in range(num_videos):
                random.shuffle(list_shuffle[n])
            list_c = [0]*num_videos
            # print "randomizing again"
        yield img, mask


