import math
import numpy as np
import time
from scipy import special
import h5py
from tuncat.run_TUnCaT import run_TUnCaT
import os
from scipy.io import loadmat


def generate_masks(network_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
        It calculates the traces of each GT neuron in "file_mask", 
        and uses TUnCaT to decontaminate the traces. 
        Then it normalizes the decontaminated traces to SNR traces. 
        For each "thred_ratio" in "list_thred_ratio", when the SNR is larger than "thred_ratio", 
        the neuron is considered active at this frame.
        For each frame, it addes all the active neurons to generate the binary temporal masks,
        and save the temporal masks in "dir_save". 

    Inputs: 
        network_input (3D numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing.
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list of float): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
        In addition, the raw and unmixed traces before and after TUnCaT are saved in the same folder
            but a different sub-folder in another "(Exp_ID).h5" file. The ".h5" file has two datasets, 
            "raw_traces" and "unmixed_traces" saving the traces before and after TUnCaT, respectively. 
    '''
    try: # If the ".mat" file is saved in '-v7.3' format
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')
        mat.close()
    except OSError: # If the ".mat" file is not saved in '-v7.3' format
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0]).astype('bool')
    (nframesf, rowspad, colspad) = network_input.shape
    (ncells, rows, cols) = rois.shape
    # The lateral shape of "network_input" can be larger than that of "rois" due to padding in pre-processing
    # This step crop "network_input" to match the shape of "rois"
    network_input = network_input[:, :rows, :cols]

    # Use TUnCaT to calculate the decontaminated traces of neural activities. 
    folder_TUnCaT = os.path.join(dir_save, 'TUnCaT')    
    start = time.time()
    (unmixed_traces, _, traces, bgtraces) = run_TUnCaT(
        Exp_ID, network_input, rois, folder_TUnCaT, list_alpha=[1], \
        Qclip=0, th_pertmin=1, epsilon=0, th_residual=0, nbin=1, \
        bin_option='downsample', multi_alpha=True, flexible_alpha=True)
    raw_traces = traces - bgtraces
    finish = time.time()
    print('TUnCaT time: {} s'.format(finish-start))

    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because TUnCaT unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=0)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=0)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=0)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[t,:]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(os.path.join(dir_temporal_masks, Exp_ID+".h5"), "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))


def generate_masks_from_traces(file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
        Similar to "generate_masks", but this version uses the traces saved in folder "traces", 
        a previous output of "generate_masks", so it does not redo TUnCaT and does not need input video.

    Inputs: 
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
    '''
    try: # If the ".mat" file is saved in '-v7.3' format
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')
        mat.close()
    except OSError: # If the ".mat" file is not saved in '-v7.3' format
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0])
    (_, rows, cols) = rois.shape
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% Extract raw and unmixed traces from the saved ".h5" file
    dir_trace = os.path.join(dir_save, "TUnCaT")
    f_raw = loadmat(os.path.join(dir_trace, 'raw', Exp_ID+".mat"))
    raw_traces = np.array(f_raw["traces"]) - np.array(f_raw["bgtraces"])
    # f_raw.close()
    f_unmix = loadmat(os.path.join(dir_trace, 'alpha= 1.000', Exp_ID+".mat"))
    unmixed_traces = np.array(f_unmix["traces_nmfdemix"])
    # f_unmix.close()
    nframesf = unmixed_traces.shape[0]
    
    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because TUnCaT unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=0)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=0)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=0)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[t,:]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(os.path.join(dir_temporal_masks, Exp_ID+".h5"), "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))

