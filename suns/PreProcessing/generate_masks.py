import math
import numpy as np
import time
from scipy import special
import h5py
import fissa
import os
from scipy.io import loadmat


def generate_masks(network_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
        It calculates the traces of each GT neuron in "file_mask", 
        and uses FISSA to decontaminate the traces. 
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
        In addition, the raw and unmixed traces before and after FISSA are saved in the same folder
            but a different sub-folder in another "(Exp_ID).h5" file. The ".h5" file has two datasets, 
            "raw_traces" and "unmixed_traces" saving the traces before and after FISSA, respectively. 
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

    # Use FISSA to calculate the decontaminated traces of neural activities. 
    folder_FISSA = os.path.join(dir_save, 'FISSA')    
    start = time.time()
    experiment = fissa.Experiment([network_input], [rois.tolist()], folder_FISSA, ncores_preparation=1)
    experiment.separation_prep(redo=True)
    prep = time.time()
    
    experiment.separate(redo_prep=False, redo_sep=True)
    finish = time.time()
    experiment.save_to_matlab()
    del network_input
    print('FISSA time: {} s'.format(finish-start))
    print('    Preparation time: {} s'.format(prep-start))
    print('    Separation time: {} s'.format(finish-prep))

    # %% Extract raw and unmixed traces from the output of FISSA
    raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
    del experiment

    # Save the raw and unmixed traces into a ".h5" file under folder "dir_trace".
    dir_trace = os.path.join(dir_save, "traces")
    if not os.path.exists(dir_trace):
        os.makedirs(dir_trace)        
    f = h5py.File(os.path.join(dir_trace, Exp_ID+".h5"), "w")
    f.create_dataset("raw_traces", data = raw_traces)
    f.create_dataset("unmixed_traces", data = unmixed_traces)
    f.close()

    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
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
        a previous output of "generate_masks", so it does not redo FISSA and does not need input video.

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
    dir_trace = os.path.join(dir_save, "traces")
    f = h5py.File(os.path.join(dir_trace, Exp_ID+".h5"), "r")
    raw_traces = np.array(f["raw_traces"])
    unmixed_traces = np.array(f["unmixed_traces"])
    f.close()
    nframesf = unmixed_traces.shape[1]
    
    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
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

