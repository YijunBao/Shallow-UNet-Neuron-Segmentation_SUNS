import random
import time
import numpy as np
import cv2
import tensorflow as tf
from unet4 import get_unet
import os

if __name__ == '__main__':
    fff = get_unet()
    fff.load_weights('Model_threshsd5.h5')
    froot = 'C:\\Matlab Files\\CNN Data\\trains and masks bin5\\trains and masks thred_ratio=5\\'
    img_folder = froot + 'test_images'
    mask_folder = froot + 'test_masks'
    n = os.listdir(img_folder)  # List of training images


    test_img = np.zeros((400, 512, 512, 1), dtype='uint8')
    test_mask = np.zeros((400, 512, 512, 1), dtype='uint8')

    for i in range(0,100):
        test_img[i,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        test_mask[i,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        test_img[i+100,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        test_mask[i+100,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        test_img[i+200,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        test_mask[i+200,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        test_img[i+300,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        test_mask[i+300,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        # test_img[i+400,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        # test_mask[i+400,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        # test_img[i+500,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        # test_mask[i+500,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        # test_img[i+600,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        # test_mask[i+600,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')
        # test_img[i+700,:487,:487,0] = cv2.imread(img_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE)
        # test_mask[i+700,:487,:487,0] = (cv2.imread(mask_folder + '/' + n[i], cv2.IMREAD_GRAYSCALE) > 100).astype('uint8')

    fff.evaluate(test_img[0:100],test_mask[0:100],batch_size=100)

    fff.evaluate(test_img,test_mask,batch_size=100)

    #for i in range(1,200):
    #test = np.expand_dims(test_img[1], axis=0)


    t = time.time()
    pp = fff.predict(test_img,batch_size=100)
    elapsed = time.time() -  t
    print(elapsed/test_img.shape[0])
    threshold = 0.93

    pp1 = np.multiply((pp[0:200] > threshold).astype('float'), test_mask[0:200])
    pp2 = (np.add((pp[0:200] > threshold).astype('float'), test_mask[0:200]) > threshold)

    IOU = np.zeros(200).astype('float')
    for i in range(0, 199):
        IOU[i] = np.sum(pp1[i])/np.sum(pp2[i])

    print(np.mean(IOU))
    print(np.std(IOU))

    #print(pp.shape)
    #print(pp.dtype)
    pp = (np.squeeze(pp[1])*255).astype(np.uint8)
    cv2.imwrite("val_out.png",pp)

