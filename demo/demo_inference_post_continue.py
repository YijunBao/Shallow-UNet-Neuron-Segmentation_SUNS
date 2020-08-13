# %%
import sys
import os
import random
import time
import glob
import numpy as np
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '..\\Network')
sys.path.insert(1, '..\\neuron_post')
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from shallow_unet import get_shallow_unet
from par2 import fastuint, fastcopy
from evaluate_post import GetPerformance_Jaccard_2
from complete_post import paremter_optimization_after


def parameter_optimization_pipeline(file_CNN, network_input, dims, \
        Params_set, filename_GT, batch_size_eval=1, useWT=False, p=None):
    # load CNN model
    (Lx, Ly) = dims
    fff = get_shallow_unet()
    fff.load_weights(file_CNN)

    # # run CNN inference once to warm up
    # fff.evaluate(network_input[:batch_size_eval],test_masks[:batch_size_eval],batch_size=batch_size_eval)
    # time_init = time.time()
    # print('Initialization: {} s'.format(time_init-start))

    # CNN inference
    start_test = time.time()
    prob_map = fff.predict(network_input, batch_size=batch_size_eval)
    finish_test = time.time()
    Time_frame = (finish_test-start_test)/network_input.shape[0]*1000
    print('Average infrence time {} ms/frame'.format(Time_frame))

    # convert the output probability map from float to uint8 to speed up future parameter optimization
    prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
    pmaps = np.zeros(prob_map.shape, dtype='uint8')
    fastuint(prob_map, pmaps)
    del prob_map, fff

    # calculate the recall, precision, and F1 when different post-processing hyper-parameters are used.
    list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT, p=p)
    return list_Recall, list_Precision, list_F1


def parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        dims, dims1, dir_img, weights_path, dir_GTMasks, dir_temp, dir_output, \
            batch_size_eval=1, useWT=False, load_exist=False):
    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    if cross_validation == "leave_one_out":
        nvideo_train = nvideo-1
        # nvideo_val = 1
    else: # cross_validation == "train_1_test_rest"
        nvideo_train = 1
        # nvideo_val = nvideo-1

    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    thresh_mask = Params_set['thresh_mask']
    list_cons = Params_set['list_cons']

    size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
    # arrays to save the recall, precision, and F1 when different post-processing hyper-parameters are used.
    F1_train = np.zeros(size_F1)
    Recall_train = np.zeros(size_F1)
    Precision_train = np.zeros(size_F1)
    (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
        =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
        # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

    
    # %% start parameter optimization for each video with various CNN models
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        list_saved_results = glob.glob(dir_temp+'Parameter Optimization * Exp{}.mat'.format(Exp_ID))
        p = mp.Pool(mp.cpu_count())
        if len(list_saved_results)<nvideo_train or not load_exist: # load SNR videos as "network_input"
            network_input = 0
            print('Video '+Exp_ID)
            start = time.time()
            h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
            nframes = h5_img['network_input'].shape[0]
            network_input = np.zeros((nframes, rows, cols, 1), dtype='float32')
            for t in range(nframes):
                network_input[t, :,:,0] = np.array(h5_img['network_input'][t])
            h5_img.close()
            # nmask = 100
            # test_masks = np.zeros((nmask, rows, cols, 1), dtype='float32')
            time_load = time.time()
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            print('Load data: {} s'.format(time_load-start))

        if cross_validation == "leave_one_out":
            list_CV = list(range(nvideo))
            list_CV.pop(eid)
        else: # cross_validation == "train_1_test_rest"
            list_CV = [eid]

        for CV in list_CV:
            mat_filename = dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID)
            if os.path.exists(mat_filename) and load_exist: # if the temporary output file already exists, load it
                mdict = loadmat(mat_filename)
                Recall_train[CV,eid] = np.array(mdict['list_Recall'])
                Precision_train[CV,eid] = np.array(mdict['list_Precision'])
                F1_train[CV,eid] = np.array(mdict['list_F1'])
        
            else:
                start = time.time()
                file_CNN = weights_path+'Model_CV{}.h5'.format(CV)
                list_Recall, list_Precision, list_F1 = parameter_optimization_pipeline(
                    file_CNN, network_input, (Lx,Ly), Params_set, filename_GT, batch_size_eval, useWT=useWT, p=p)
                # # load CNN model
                # fff = get_shallow_unet()
                # fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))

                # # run CNN inference once to warm up
                # fff.evaluate(network_input[:nmask],test_masks,batch_size=batch_size_eval)
                # time_init = time.time()
                # print('Initialization: {} s'.format(time_init-start))

                # # CNN inference
                # start_test = time.time()
                # prob_map = fff.predict(network_input, batch_size=batch_size_eval)
                # finish_test = time.time()
                # Time_frame = (finish_test-start_test)/network_input.shape[0]*1000
                # print('Average infrence time {} ms/frame'.format(Time_frame))

                # # convert the output probability map from float to uint8 to speed up future parameter optimization
                # prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
                # pmaps = np.zeros(prob_map.shape, dtype='uint8')
                # fastuint(prob_map, pmaps)
                # del prob_map, fff
        
                # # calculate the recall, precision, and F1 when different post-processing hyper-parameters are used.
                # list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT, p=p)
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                
                Recall_train[CV,eid] = list_Recall
                Precision_train[CV,eid] = list_Precision
                F1_train[CV,eid] = list_F1
                # save recall, precision, and F1 in a temporary ".mat" file
                mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
                savemat(mat_filename, mdict) 
                # del pmaps

        p.close()
        # del network_input
            
            
    # %%
    for CV in range(nvideo):
        # calculate the mean recall, precision, and F1 of all the training videos
        Recall_mean = Recall_train[CV].mean(axis=0)*nvideo/nvideo_train
        Precision_mean = Precision_train[CV].mean(axis=0)*nvideo/nvideo_train
        F1_mean = F1_train[CV].mean(axis=0)*nvideo/nvideo_train
        Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
            array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
        print('F1_max=', [x.max() for x in F1_train[CV]])

        # find the post-processing hyper-parameters to achieve the highest average F1 over the training videos
        ind = F1_mean.argmax()
        ind = np.unravel_index(ind,F1_mean.shape)
        minArea = list_minArea[ind[0]]
        avgArea = list_avgArea[ind[1]]
        thresh_pmap = list_thresh_pmap[ind[2]]
        thresh_COM = list_thresh_COM[ind[3]]
        thresh_IOU = list_thresh_IOU[ind[4]]
        thresh_consume = (1+thresh_IOU)/2
        cons = list_cons[ind[5]]
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        print('F1_mean=', F1_mean[ind])

        # save the optimal hyper-parameters a ".mat" file
        Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
            'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
        savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)


# %%
if __name__ == '__main__':
    # %% setting parameters
    rate_hz = 10 # frame rate of the video
    Dimens = (120,88) # lateral dimensions of the video
    nframes = 3000 # number of frames for each video
    Mag = 6/8 # spatial magnification compared to ABO videos.

    batch_size_eval = 100 # batch size in CNN inference
    useWT=False # True if using additional watershed
    load_exist=False # True if using temp files already saved in the folders

    (rows, cols) = Dimens # size of the network input and output
    (Lx, Ly) = (rows, cols) # size of the original video

    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = 'data\\' 
    # Cross-validation strategy. Can be "leave_one_out" or "train_1_test_rest"
    cross_validation = "leave_one_out"
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 

    dir_parent = dir_video + 'complete\\' # folder to save all the processed data
    dir_sub = ''
    dir_img = dir_parent + 'network_input\\' # folder of the SNR videos
    weights_path = dir_parent + dir_sub + 'Weights\\' # folder of the trained CNN
    dir_output = dir_parent + dir_sub + 'output_masks\\' # folder to save the optimized hyper-parameters
    dir_temp = dir_parent + dir_sub + 'temp\\' # temporary folder to save the F1 with various hyper-parameters

    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    # %% set the range of hyper-parameters to be optimized in
    # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
    list_minArea = list(range(30,85,5)) 
    # average area of a typical neuron (unit: pixels in ABO videos)
    list_avgArea = [177] 
    # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1.5)
    list_thresh_pmap = list(range(130,235,10))
    # threshold to binarize the neuron masks. For each mask, 
    # values higher than "thresh_mask" times the maximum value of the mask are set to one.
    thresh_mask = 0.5
    # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
    thresh_COM0 = 2
    # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
    list_thresh_COM = list(np.arange(4, 9, 1)) 
    # minimum IoU of two masks to be considered the same neuron
    list_thresh_IOU = [0.5] 
    # minimum consecutive number of frames of active neurons
    list_cons = list(range(1, 8, 1)) 

    # adjust the units of the hyper-parameters to pixels in the test videos according to relative magnification
    list_minArea= list(np.round(np.array(list_minArea) * Mag**2))
    list_avgArea= list(np.round(np.array(list_avgArea) * Mag**2))
    thresh_COM0= thresh_COM0 * Mag
    list_thresh_COM= list(np.array(list_thresh_COM) * Mag)
    # adjust the minimum consecutive number of frames according to different frames rates between ABO videos and the test videos
    # list_cons=list(np.round(np.array(list_cons) * rate_hz/30).astype('int'))

    # dictionary of all fixed and searched post-processing parameters.
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
            'thresh_mask': thresh_mask, 'list_cons': list_cons}
    print(Params_set)

    parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        (Lx, Ly), (rows, cols), dir_img, weights_path, dir_GTMasks, dir_temp, dir_output, \
        batch_size_eval, useWT=useWT, load_exist=load_exist)
    # nvideo = len(list_Exp_ID) # number of videos used for cross validation
    # if cross_validation == "leave_one_out":
    #     nvideo_train = nvideo-1
    #     nvideo_val = 1
    # else: # cross_validation == "train_1_test_rest"
    #     nvideo_train = 1
    #     nvideo_val = nvideo-1

    # size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
    # # arrays to save the recall, precision, and F1 when different post-processing hyper-parameters are used.
    # F1_train = np.zeros(size_F1)
    # Recall_train = np.zeros(size_F1)
    # Precision_train = np.zeros(size_F1)
    # (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
    #     =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
    #     # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

    
    # # %% start parameter optimization
    # for (eid,Exp_ID) in enumerate(list_Exp_ID):
    #     list_saved_results = glob.glob(dir_temp+'Parameter Optimization * Exp{}.mat'.format(Exp_ID))
    #     p = mp.Pool(mp.cpu_count())
    #     if len(list_saved_results)<nvideo_train or not load_exist: # load SNR videos as "network_input"
    #         network_input = 0
    #         print('Video '+Exp_ID)
    #         start = time.time()
    #         h5_img = h5py.File(dir_img+Exp_ID+'.h5', 'r')
    #         nframes = h5_img['network_input'].shape[0]
    #         network_input = np.zeros((nframes, rows, cols, 1), dtype='float32')
    #         for t in range(nframes):
    #             network_input[t, :,:,0] = np.array(h5_img['network_input'][t])
    #         h5_img.close()
    #         nmask = 100
    #         test_masks = np.zeros((nmask, rows, cols, 1), dtype='float32')
    #         time_load = time.time()
    #         filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
    #         print('Load data: {} s'.format(time_load-start))

    #     if cross_validation == "leave_one_out":
    #         list_CV = list(range(nvideo))
    #         list_CV.pop(eid)
    #     else: # cross_validation == "train_1_test_rest"
    #         list_CV = [eid]

    #     for CV in list_CV:
    #         mat_filename = dir_temp+'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID)
    #         if os.path.exists(mat_filename) and load_exist: # if the temporary output file already exists, load it
    #             mdict = loadmat(mat_filename)
    #             Recall_train[CV,eid] = np.array(mdict['list_Recall'])
    #             Precision_train[CV,eid] = np.array(mdict['list_Precision'])
    #             F1_train[CV,eid] = np.array(mdict['list_F1'])
        
    #         else:
    #             start = time.time()
    #             file_CNN = weights_path+'Model_CV{}.h5'.format(CV)
    #             list_Recall, list_Precision, list_F1 = parameter_optimization_pipeline(
    #                 file_CNN, network_input, (Lx,Ly), Params_set, filename_GT, batch_size_eval, useWT=useWT, p=p)
    #             # # load CNN model
    #             # fff = get_shallow_unet()
    #             # fff.load_weights(weights_path+'Model_CV{}.h5'.format(CV))

    #             # # run CNN inference once to warm up
    #             # fff.evaluate(network_input[:nmask],test_masks,batch_size=batch_size_eval)
    #             # time_init = time.time()
    #             # print('Initialization: {} s'.format(time_init-start))

    #             # # CNN inference
    #             # start_test = time.time()
    #             # prob_map = fff.predict(network_input, batch_size=batch_size_eval)
    #             # finish_test = time.time()
    #             # Time_frame = (finish_test-start_test)/network_input.shape[0]*1000
    #             # print('Average infrence time {} ms/frame'.format(Time_frame))

    #             # # convert the output probability map from float to uint8 to speed up future parameter optimization
    #             # prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
    #             # pmaps = np.zeros(prob_map.shape, dtype='uint8')
    #             # fastuint(prob_map, pmaps)
    #             # del prob_map, fff
        
    #             # # calculate the recall, precision, and F1 when different post-processing hyper-parameters are used.
    #             # list_Recall, list_Precision, list_F1 = paremter_optimization_after(pmaps, Params_set, filename_GT, useWT=useWT, p=p)
    #             Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
    #                 array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                
    #             Recall_train[CV,eid] = list_Recall
    #             Precision_train[CV,eid] = list_Precision
    #             F1_train[CV,eid] = list_F1
    #             # save recall, precision, and F1 in a temporary ".mat" file
    #             mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
    #             savemat(mat_filename, mdict) 
    #             # del pmaps

    #     p.close()
    #     # del network_input
            
            
    # # %%
    # for CV in range(nvideo):
    #     # calculate the mean recall, precision, and F1 of all the training videos
    #     Recall_mean = Recall_train[CV].mean(axis=0)*nvideo/nvideo_train
    #     Precision_mean = Precision_train[CV].mean(axis=0)*nvideo/nvideo_train
    #     F1_mean = F1_train[CV].mean(axis=0)*nvideo/nvideo_train
    #     Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
    #         array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
    #     print('F1_max=', [x.max() for x in F1_train[CV]])

    #     # find the post-processing hyper-parameters to achieve the highest average F1 over the training videos
    #     ind = F1_mean.argmax()
    #     ind = np.unravel_index(ind,F1_mean.shape)
    #     minArea = list_minArea[ind[0]]
    #     avgArea = list_avgArea[ind[1]]
    #     thresh_pmap = list_thresh_pmap[ind[2]]
    #     thresh_COM = list_thresh_COM[ind[3]]
    #     thresh_IOU = list_thresh_IOU[ind[4]]
    #     thresh_consume = (1+thresh_IOU)/2
    #     cons = list_cons[ind[5]]
    #     Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
    #         'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
    #     print(Params)
    #     print('F1_mean=', F1_mean[ind])

    #     # save the optimal hyper-parameters a ".mat" file
    #     Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
    #         'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
    #     savemat(dir_output+'Optimization_Info_{}.mat'.format(CV), Info_dict)

