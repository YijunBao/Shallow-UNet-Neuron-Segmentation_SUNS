![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# SUNS
Shallow UNet Neuron Segmentation (SUNS) is an automatic algorithm to segment active neurons from two-photon calcium imaging videos. It used temporal filtering and whitening schemes to extract temporal features associated with active neurons, and used a compact shallow U-Net to extract spatial features of neurons.

Copyright (C) 2020 Duke University NeuroToolbox

![Example video](readme/Masks%202%20raw.gif)

- [SUNS](#suns)
- [System requirement](#system-requirement)
- [Installation on Windows or Linux](#installation-on-windows-or-linux)
- [Demo](#demo)
- [Input, Output, and Intermediate Files](#input-output-and-intermediate-files)
  - [Input files](#input-files)
  - [Output files](#output-files)
  - [Intermediate files](#intermediate-files)
- [Use your own data](#use-your-own-data)
  - [Use your own videos](#use-your-own-videos)
  - [Use your own GT masks](#use-your-own-gt-masks)
  - [Use your own temporal filter kernel](#use-your-own-temporal-filter-kernel)
  - [No training data?](#no-training-data)
  - [Use optional spatial filtering?](#use-optional-spatial-filtering)
- [Links to Datasets and Manual Markings](#links-to-datasets-and-manual-markings)
- [Citing](#citing)
- [Licensing and Copyright](#licensing-and-copyright)
- [Sponsors](#sponsors)


# System requirement
* Memory: ~6x file size of the raw video if the the raw video is in uint16 format. ~3x file size of the raw video if the the raw video is in float32 format. 
* A CUDA compatible GPU is preferred.


# Installation on Windows or Linux
* Install [Anaconda](https://www.anaconda.com/)
* Launch Anaconda prompt and type the following in order (SUNS_python_root_path is the directory to which the provided files were downloaded to, such as `C:/Users/(username)/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS`):
```bat
cd SUNS_python_root_path
cd installation
conda env create -f environment_suns.yml -n suns
```
* Go to the Anaconda environment foler, (such as `C:/ProgramData/Anaconda3/envs` or `C:/Users/(username)/.conda/envs`), and then go to folder `suns/Lib/site-packages/fissa`, overwrite `core.py` with the files provided in the `installation` folder. The modified files increase speed by eliminating redundant `separate` or `separation_prep` during initializating an `Experiment` object, and enable videos whose size are larger than 4 GB after converting to float32. If neither of them is important to you, then you can skip replacing the files. If you see a lot of text output when activating suns environment and do not want to see them, you can go to the Anaconda environment foler, go to folder `suns/etc/conda/activate.d`, and delete the two files under this folder. 

The installation should take less than half an hour in total. The first run of the software may take some additional time (up to 20 minutes on a laptop) to add the GPU, but this extra time will not occur in later runs.


# Demo
We provided a demo for all users to get familiar with our software. We provided four two-photon imaging videos as well as their manually marked neurons in `demo/data`, adapted from [CaImAn dataset](https://zenodo.org/record/1659149). The demo will perform a cross validation over the four videos: train the CNN and search for optimal hyper-parameters using three videos, and test SUNS with the training output on the remaining video. 

To run the demo on Windows, launch Anaconda prompt and type the following script 
```bat
cd SUNS_python_root_path
cd demo
conda activate suns
.\demo_pipeline.bat
```

The demo contains four parts: training CNN and hyper-parameters, testing SUNS batch, testing SUNS online, and testing SUNS online with tracking. The input, output, and intermediate files will be explained in [Input, Output, and Intermediate Files](#input-output-and-intermediate-files). Alternatively, you can also run `demo_pipeline_1to3.bat` instead of `demo_pipeline.bat`. The pipeline `demo_pipeline.bat` does standard leave-one-out cross validation on the four example videos. The pipeline `demo_pipeline_1to3.bat` trains the CNN model and post-processing parameters on one video, and tests on the remaining videos. The expected average F1 scores and example processing time on Windows are shown on the following tables.

Expected average F1 score
|	|Train	|Batch	|Online	|Track|
|:------:|:------:|:------:|:------:|:------:|
|train 3 test 1	|0.77	|0.75	|0.67	|0.65|
|train 1 test 3	|0.81	|0.72	|0.66	|0.66|

Example running time
|CPU	|GPU	|Train 3 to 1<br>(total)	|Train 1 to 3<br>(total)	|Batch<br>(average)	|Online<br>average)	|Track<br>(average)|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|AMD 1920X<br>12-core	|NVIDIA Titan RTX|1.0 h	|0.4 h	|1.5 s	|18 s	|20s	|
|Intel i7-6800K<br>6-core	|NVIDIA GTX 1080|1.5 h	|0.6 h	|1.7 s	|16 s	|20 s	|
|Intel i5-6200U<br>dual-core	|NVIDIA 940MX	|5.1 h	|1.8 h	|4.3 s	|24 s	|28 s|


To run the demo on Linux, launch Anaconda prompt and type the following script 
```sh
cd SUNS_python_root_path
cd demo
conda activate suns
sh demo_pipeline.sh
```
If this does not run through due to multiprocessing errors, another option is to copy the files in the folder `demo/split_version` to the `demo` folder and run `demo_pipeline_split.sh` instead of `demo_pipeline.sh`.

Due to the randomness in training CNN, there is a low chance (~1%) that the trained CNN model is bad (e.g., the resulting F1 score is 0). If that happens, simply retrain the CNN one more time, and it should work out.


# Input, Output, and Intermediate Files
When you download this repo, you will see some files and a folder under `demo/data`. All the input, output, and intermediate files are saved under this folder. 

## Input files
* The .h5 files are the input videos contained in dataset 'mov' (shape = (3000, 120, 88)). 
* The `GT Masks` folder contains the ground truth masks of each video. `FinalMasks_YST_part??.mat` stores the GT masks in a 3D array (shape = (88, 120, n) in MATLAB), and `FinalMasks_YST_part??_sparse.mat` stores the GT masks in a 2D sparse matrix (shape = (88*120, n) in MATLAB). 

## Output files
After running a complete pipeline, the intermediate and output files will be under a new folder `demo/data/noSF`. 
* `output_masks` stores the output masks of SUNS batch. 
* `output_masks online` stores the output masks of SUNS online. 
* `output_masks track` stores the output masks of SUNS online with tracking. 

The segmented masks are stored in `Output_Masks_{}.mat` as a 3D array with shape (n, 120, 88) in MATLAB, so you may need to transpose the demensions to match the GT masks. <br>
The speed and accuracy scores are stored in the same three folders as `Output_Info_All.mat`. 

## Intermediate files
After running a complete pipeline, the intermediate and output files will be under a new folder `demo/data/noSF`. Most of intermediate files are numbered according to the video name and/or the index of cross validation. 
* `network_input` stores the SNR videos after pre-processing (in dataset 'network_input').
* `FISSA` stores the temporary output of [FISSA](https://github.com/rochefort-lab/fissa).
* `traces` stores the traces of the GT neurons after decontamination using FISSA.
* `temporal_masks(3)` stores the temporal masks of active neurons (in dataset 'temporal_masks')used to train CNN.
* `Weights` stores the trained CNN model.
* `training output` stores the loss after each training epoch.
* `temp` stores the recall, precision, and F1 score of all parameter combinations for each video and each cross validation.
* `output_masks` stores the optimal hyper-parameters in `Optimization_Info_{}.mat`. In this file, `Params_set` stores the searching ranges of the hyper-parameters, and `Params` stores the optimized hyper-parameters. 

# Use your own data
Of course, you can modify the demo scripts to process other videos. You need to change the parameters in the python scripts (e.g., dimension of each frame, spatial magnification, number of frames, recording frame rate) to correspond to your videos. When you search for optimal hyper-parameters, you also need to change the search ranges in `demo_train_CNN_params.py`. Followings are more instructions. 

## Use your own videos
* Set the folder and file names of your video correctly. You need to change the variable `dir_video` (folder of the input videos) and `list_Exp_ID` (list of file names of the input videos) in all the python scripts. 
* Currently, we only support .h5 files as the input video, so you need to convert the format of your data to .h5, and save the video in dataset 'mov'. The video should have a shape of (T, Lx, Ly), where T is the number of frames, and (Lx, Ly) is the lateral dimension of each frame. The support to more video formats will come soon. 

## Use your own GT masks
* If you want to train your own network, create a folder `GT Masks` under the video folder, and copy all the ground truth markings into that folder. 
* Currently, we only support .mat files as the GT masks, so you need to convert the format of your GT masks to .mat, and save the GT masks in dataset 'FinalMasks'. The name of the masks file should be `FinalMasks_{Exp_ID}.mat`, where the `{Exp_ID}` is the name of the corresponding video. The GT masks should be saved as a 3D array named `FinalMasks`, and the dimension should be (Ly, Lx, n) in MATLAB, where Lx and Ly are the same as the lateral dimension of the video, and n is the number of masks.
* Because we mainly use sparse matrix representations of the masks to speed up post-processing, you also need to convert the masks from a 3D array to a 2D sparse matrix. You can modify the MATLAB script `utils/generate_sparse_GT.m` to do that. 

## Use your own temporal filter kernel
* In the demo, we used `demo/YST_spike_tempolate.h5` to store the matched filter kernel for our temporal filtering. We determined the matched filter kernel by averaging calcium transients within a moderate SNR range from the videos. The filter kernel was generated by the MATLAB scripts `utils/temporal filter/calculate_traces_bgtraces_demo.m` and `utils/temporal filter/temporal_filter_demo.m`. You can modify these scripts to generate the matched filter kernel for your dataset, and you need to change the variable `filename_TF_template` in all the python scripts to your filter file. 
* Alternatively, you can also use a simple exponential or double exponential curve (with appropriate decay time constants) as the filter kernel. We provided an example exponentially decay kernel in comments of the demo scripts. 
* If the quality of your video is very good, and the signal to noise ratio is high, it is OK to not use temporal filtering. If you decide not to use temporal filtering, you can set `useTF=False` in all python scripts. 

## No training data?
* SUNS requires a CNN model and post-processing parameters to process a video. It is best to train your own CNN models and post-processing parameters for your dataset, because the optimal CNN models and post-processing parameters, especially parameters, can be different with different imaging conditions. Therefore, our expectation is that you manually label at least one video recorded by your two-photon microscope, and use that to train the CNN and parameters. 
* However, at the beginning, you can also use the CNN model in `training results/Weights/Model_CV10.h5` and the parameters in `training results/output_masks/Optimization_Info_10.mat`. You can do so by changing `filename_CNN` (the path of the CNN model) and `Optimization_Info` (the path of the optimal post-processing parameters) in the demo script. We don't recommnd using the training results from the demo videos, because the demo videos are relatively small. Our recommended CNN model and hyper-parameters were trained from 10 large videos, so they should perform better in general. 

## Use optional spatial filtering?
We have an optional spatial filtering step in the pre-processing. Althought it did not improve the performance in many of our experiements, it sometimes can improve the accuracy. If you want to try spatial filtering, you need to set `useSF=True` in all python scripts. We recommend you to uncommend the first block of the `demo_pipeline.bat` and run `demo_learn_wisdom.py` and `demo_learn_wisdom_2d.py` (change the dimensions in these scripts to fit your videos before running). After running them, you will find some text files under `demo/wistom`, which stores the learned wisdom used for spatial filtering. When you have these files with correct dimensions of your video, the preparation of spatial filtering will be faster.


# Links to Datasets and Manual Markings
In our paper, we used two-photon imaging videos from [Allen Brain Observatory dataset](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS), [Neurofinder Challenge website](https://github.com/codeneuro/neurofinder), and [CaImAn dataset](https://zenodo.org/record/1659149). We used the manual markings of Allen Brain Observatory and Neurofinder from [STNeuroNet](https://github.com/soltanianzadeh/STNeuroNet) repository, and used the manual markings of CaImAn dataset from [CaImAn dataset](https://zenodo.org/record/1659149). A more detailed instruction is given under the folder `paper reproduction`. 


# Citing 
If you use any part of this software in your work, please cite Bao et al. 2020:


# Licensing and Copyright
SUNS is released under [the GNU License, Version 2.0](LICENSE).


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
