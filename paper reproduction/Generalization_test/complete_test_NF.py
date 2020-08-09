# %%
import os
# import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import sys
from scipy import sparse

from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(0, '..\\PreProcessing')
sys.path.insert(0, '..\\Network')
sys.path.insert(0, '..\\neuron_post')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import par1
# from preprocessing_functions import process_video, process_video_prealloc
# from par2 import fastuint, fastcopy
from par3 import fastthreshold
from unet4_best import get_unet
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import complete_segment
from functions_other_data import process_video_prealloc_others, data_info_neurofinder, temporal_filter

# %%
if __name__ == '__main__':
    list_neurofinder = ['01.00', '01.01', '02.00', '02.01', '04.00', '04.01']
    for trainset_type in {'train'}: #  ,'test'
        testset_type = list({'train','test'}-{trainset_type})[0]
        nvideo = len(list_neurofinder)

        thred_std = 3
        num_train_per = 1800
        BATCH_SIZE = 20
        NO_OF_EPOCHS = 200
        # frames_init = 300
        # merge_every = 10
        useSF=False
        useTF=True
        useSNR=True
        useWT=False
        batch_size_eval = 200

        dir_video_test = 'E:\\NeuroFinder\\{} videos\\'.format(testset_type)
        dir_video_train = 'E:\\NeuroFinder\\{} videos\\'.format(trainset_type)
        dir_parent_test = dir_video_test + 'ShallowUNet\\noSF\\'
        dir_parent_train = dir_video_train + 'ShallowUNet\\noSF\\'
        dir_sub = '1to1\\DL+BCE\\std{}_nf{}_ne{}_bs{}\\'.format(thred_std, num_train_per, NO_OF_EPOCHS, BATCH_SIZE)
        weights_path = dir_parent_train + dir_sub + 'train\\Weights\\'
        dir_params = dir_parent_train + dir_sub + 'train\\params\\' #
        dir_output = dir_parent_test + dir_sub + 'test\\output_masks complete\\'
        dir_GTMasks = dir_video_test + 'GTMasks_'

        if not os.path.exists(dir_output):
            os.makedirs(dir_output) 

        # %% PreProcessing parameters
        nn = 10000 # nframes # cv2.getOptimalDFTSize(nframes)
        gauss_filt_size = 50  # signa in pixels
        num_median_approx = 1000
        network_baseline = 0 # 64.0
        network_SNRscale = 1 # 32.0
        # frame_rate = 30
        # decay = 0.2
        # leng_tf = math.ceil(frame_rate*decay) # 6
        # h5f = h5py.File(r'C:\Matlab Files\PreProcessing\GCaMP6f_spike_tempolate_mean.h5','r')
        # Poisson_filt = np.array(h5f['filter_tempolate']).squeeze()[3:12].astype('float32')

        # %% PostProcessing parameters
        # Lx = Ly = 512
        p = mp.Pool() #mp.cpu_count()

        list_vid = list(range(0,6))
        num_CV = len(list_vid)
        list_Recall = np.zeros((num_CV, 1))
        list_Precision = np.zeros((num_CV, 1))
        list_F1 = np.zeros((num_CV, 1))
        list_time = np.zeros((num_CV, 4))
        list_time_frame = np.zeros((num_CV, 4))

        for vid in list_vid: #[5:6]
            # if vid!=5:
            #     continue
            Exp_ID = list_neurofinder[vid]
            if trainset_type == 'test':
                Exp_ID_train = Exp_ID + '.test'
            else:
                Exp_ID_train = Exp_ID
            if testset_type == 'test':
                Exp_ID_test = Exp_ID + '.test'
            else:
                Exp_ID_test = Exp_ID
            # if trainset_type == 'train':
            #     Exp_ID_test = Exp_ID + '.test'
            # elif trainset_type == 'test':
            #     Exp_ID_test = Exp_ID
            print('Video ', Exp_ID)
            fname_output_masks = dir_output+'Output_Masks_{}.mat'.format(Exp_ID_test)

            # if os.path.exists(fname_output_masks):
            #     start = time.time()
            #     file_Masks = loadmat(fname_output_masks)
            #     Masks = np.array(file_Masks['Masks'])
            #     dims_Masks = Masks.shape
            #     Masks_2 = sparse.csr_matrix(Masks.reshape((dims_Masks[0], dims_Masks[1]*dims_Masks[2])))
            #     finish = time.time()
            #     nframes = 1
            #     (time_pre, time_CNN, time_post, time_all) = (0,0,0,0)
            #     (time_frame_pre, time_frame_CNN, time_frame_post, time_frame_all) = (0,0,0,0)
            # else:
            fname_info = dir_video_test + 'neurofinder.' + Exp_ID_test + '\\info.json'
            info, Mxy = data_info_neurofinder(fname_info)
            dims_raw = info['dimensions'][::-1]

            start = time.time()
            fff = get_unet() #size
            fff.load_weights(weights_path+'Model_{}.h5'.format(Exp_ID_train))
            size = 512
            init_imgs = np.zeros((batch_size_eval, size, size, 1), dtype='float32')
            init_masks = np.zeros((batch_size_eval, size, size, 1), dtype='uint8')
            fff.evaluate(init_imgs, init_masks, batch_size=batch_size_eval)

            # frame_rate = info['rate-hz']
            # indicator = info['indicator']
            # # decay = info['decay_time']
            # leng_tf = np.ceil(frame_rate*decay)+1
            # Poisson_filt = np.exp(-np.arange(leng_tf)/frame_rate/decay).astype('float32')
            # Poisson_filt = Poisson_filt / Poisson_filt.sum()
            Poisson_filt = temporal_filter(info)
            # print(Poisson_filt.size)
            # continue
            Params_pre = {'gauss_filt_size':gauss_filt_size*Mxy, 'num_median_approx':num_median_approx, 
                'network_baseline':network_baseline, 'network_SNRscale':network_SNRscale, 
                'nn':nn, 'Poisson_filt': Poisson_filt} #, 'frame_rate':frame_rate, 'decay':decay, 'leng_tf':leng_tf
            Optimization_Info = loadmat(dir_params+'Optimization_Info_{}.mat'.format(Exp_ID_train))
            Params_post_mat = Optimization_Info['Params'][0]
            Params_post={'minArea': Params_post_mat['minArea'][0][0,0], 
                'avgArea': Params_post_mat['avgArea'][0][0,0],
                'thresh_pmap': Params_post_mat['thresh_pmap'][0][0,0], 
                'thresh_mask': Params_post_mat['thresh_mask'][0][0,0], 
                'thresh_COM0': Params_post_mat['thresh_COM0'][0][0,0], 
                'thresh_COM': Params_post_mat['thresh_COM'][0][0,0], 
                'thresh_IOU': Params_post_mat['thresh_IOU'][0][0,0], 
                'thresh_consume': Params_post_mat['thresh_consume'][0][0,0], 
                'cons':Params_post_mat['cons'][0][0,0]}
            thresh_pmap_float = (Params_post_mat['thresh_pmap'][0][0,0]+1.5)/256
            time_init = time.time()
            print('Initialization time: {} s'.format(time_init-start))
            print(Params_post)
            Params_post['thresh_pmap'] = None # Avoid repeated thresholding in postprocessing

            # %% Load data and preparation
            # h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
            # test_imgs = np.expand_dims(np.array(h5_img['network_input']), axis=-1)
            # h5_img.close()
            # nmask = 100
            # time_load = time.time()
            # print('Load data: {} s'.format(time_load-time_init))

            # %% PreProcessing including loading data
            # start = time.time()
            video_input, start, raw_dims = process_video_prealloc_others(dir_video_test, Exp_ID_test, \
                Params_pre, useSF=useSF, useTF=useTF, useSNR=useSNR, isonline=True)
            end_pre = time.time()
            (nframes, Lx, Ly) = video_input.shape #[0]
            time_pre = end_pre-start
            time_frame_pre = time_pre/nframes*1000
            print('PreProcessing time: {:6f} s, {:6f} ms/frame'.format(time_pre, time_frame_pre))

            # %% Network inference
            video_input = np.expand_dims(video_input, axis=-1)
            start_network = time.time()
            prob_map = fff.predict(video_input, batch_size=batch_size_eval)
            end_network = time.time()
            time_CNN = end_network-start_network
            time_frame_CNN = time_CNN/nframes*1000
            print('CNN Infrence time: {:6f} s, {:6f} ms/frame'.format(time_CNN, time_frame_CNN))

            # prob_map = prob_map.squeeze()#[:, :Lx, :Ly]
            # # # pmaps =(prob_map*256-0.5).astype(np.uint8)
            # pmaps = np.zeros(prob_map.shape, dtype='uint8')
            # fastuint(prob_map, pmaps)
            # # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
            # # fastcopy(prob_map, pmaps) 
            # # pmaps = prob_map

            prob_map = prob_map.squeeze()[:, :Lx, :Ly]
            # %% PostProcessing
            start_post = time.time()
            # # pmaps =(prob_map*256-0.5).astype(np.uint8)
            pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
            fastthreshold(prob_map, pmaps_b, thresh_pmap_float)
            # fastuint(prob_map, pmaps)
            # pmaps = np.zeros(prob_map.shape, dtype=prob_map.dtype)
            # fastcopy(prob_map, pmaps) 
            # pmaps = prob_map

            # %% PostProcessing
            start_post = time.time()
            Masks_2 = complete_segment(pmaps_b, Params_post, display=True, p=p, useWT=useWT)
            Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly))
            finish = time.time()
            time_post = finish-start_post
            time_frame_post = time_post/nframes*1000
            print('PostProcessing time: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

            # f = h5py.File(dir_output + 'probability_map {}.h5'.format(Exp_ID_test), "w")
            # f.create_dataset("probability_map", data = pmaps)
            # f.close()

            # %% Evaluation
            filename_GT = dir_GTMasks + Exp_ID_test + '_sparse.mat'
            data_GT=loadmat(filename_GT)
            GTMasks_2 = data_GT['GTMasks_2'].transpose()

            (Recall,Precision,F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
            print({'Recall':Recall, 'Precision':Precision, 'F1':F1})
            savemat(fname_output_masks, {'Masks':Masks})

            # %% Save information
            time_all = finish-start
            time_frame_all = time_all/nframes*1000
            print('Total time: {:6f} s, {:6f} ms/frame'.format(time_all, time_frame_all))
            list_Recall[vid] = Recall
            list_Precision[vid] = Precision
            list_F1[vid] = F1
            list_time[vid] = np.array([time_pre, time_CNN, time_post, time_all])
            list_time_frame[vid] = np.array([time_frame_pre, time_frame_CNN, time_frame_post, time_frame_all])

            # del video_input, pmaps, prob_map, Masks
            Info_dict = {'Params':Params_post, 'Recall':Recall, 'Precision':Precision, 'F1':F1, 
                'time':list_time[vid], 'time_frame':list_time_frame[vid]}
            savemat(dir_output+'Output_Info_{}.mat'.format(Exp_ID_test), Info_dict)
            Info_dict_all = {'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 
                'list_time':list_time, 'list_time_frame':list_time_frame}
            savemat(dir_output+'Output_Info_All.mat', Info_dict_all)

    # %% Plot some figures
            # SNR_max = video_input.squeeze().max(axis=0)
            # plt.imshow(SNR_max[:raw_dims[1], :raw_dims[2]], vmin=2, vmax=10)
            # plt.title('Max Projection of SNR Video')
            # plt.colorbar()
            # plt.savefig(dir_output + Exp_ID_test + '_SNR_max.png')
            # # plt.show()
            # plt.clf()

            # GTMasks_sum = GTMasks_2.sum(axis=0).A.reshape(Lx,Ly)
            # plt.imshow(GTMasks_sum[:raw_dims[1], :raw_dims[2]])
            # plt.title('Sum of Ground Truth Masks')
            # plt.colorbar()
            # plt.savefig(dir_output + Exp_ID_test + '_GTMasks.png')
            # # plt.show()
            # plt.clf()

            # pmaps_max = pmaps.max(axis=0)
            # plt.imshow(pmaps_max[:raw_dims[1], :raw_dims[2]])
            # plt.title('Max Projection of Probability Map')
            # plt.colorbar()
            # plt.savefig(dir_output + Exp_ID_test + '_pmaps_max.png')
            # # plt.show()
            # plt.clf()

            # Masks_sum = Masks.sum(axis=0)
            # plt.imshow(Masks_sum[:raw_dims[1], :raw_dims[2]])
            # plt.title('Sum of Segmented Masks')
            # plt.colorbar()
            # plt.savefig(dir_output + Exp_ID_test + '_Masks.png')
            # # plt.show()
            # plt.clf()

        p.close()
    # %%
