![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# SUNS
Shallow UNet Neuron Segmentation (SUNS) is an automatic algorithm to segment active neurons from two-photon calcium imaging videos. It used temporal filtering and whitening schemes to extract temporal features associated with active neurons, and used a compact shallow U-Net to extract spatial features of neurons.

Copyright (C) 2020 Duke University NeuroToolbox

![Example video](readme/Masks%202%20raw.gif)

- [SUNS](#suns)
- [Documentation](#documentation)
- [System requirement](#system-requirement)
- [Installation on Windows](#installation-on-windows)
- [Demo](#demo)
- [Links to Datasets and Manual Markings](#links-to-datasets-and-manual-markings)
- [Citing](#citing)
- [Licensing and Copyright](#licensing-and-copyright)
- [Sponsors](#sponsors)


# Documentation
The how-to guides are available on [the Wiki](https://github.com/YijunBao/Shallow-UNet-Neuron-Segmentation_SUNS/wiki).


# System requirement
* Operation system: Windows 10.
* Memory: ~6x file size of the raw video if the the raw video is in uint16 format. ~3x file size of the raw video if the the raw video is in float32 format. 
* A CUDA compatible GPU is preferred.


# Installation on Windows
* Install [Anaconda](https://www.anaconda.com/) with Python 3.7
* Launch Anaconda prompt and type the following in order (SUNS_python_root_path is the directory to which the provided files were downloaded to, such as `C:/Users/(username)/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS`):
```bat
cd SUNS_python_root_path
cd installation
conda env create -f environment_suns.yml -n suns
```
* Go to the Anaconda environment foler, (such as `C:/ProgramData/Anaconda3/envs` or `C:/Users/(username)/.conda/envs`), and then go to folder `suns/Lib/site-packages/fissa`, overwrite `core.py` and `neuropil.py` with the files provided in the `installation` folder. If the dataset you used is less than 4 GB after the data type is converted to float32 and you don't mind a lot of text output from FISSA, you can skip this step. 

The installation should take less than half an hour in total. The first run of the software may take some additional time (up to 20 minutes on a laptop) to add the GPU, but this extra time will not occur in later runs.


# Demo
We provided a demo for all users to get familiar with our software. We provided four two-photon imaging videos as well as their manually marked neurons in `demo/data`. The demo will perform a cross validation over the four videos: train the CNN and search for optimal hyper-parameters using three videos, and test SUNS with the training output on the remaining video. 

To run the demo, launch Anaconda prompt and type the following script 
```bat
cd SUNS_python_root_path
cd demo
conda activate suns
demo_pipeline.bat
```
The demo contains three parts: training CNN and hyper-parameters, testing SUNS batch, and testing SUNS online. The output masks of SUNS batch will be in `demo/complete/output_masks`, the output masks of SUNS online will be in `demo/complete/output_masks online`, and the output masks of SUNS online with tracking will be in `demo/complete/output_masks track`. The average F1 score of the training videos should be ~0.8, and the average F1 of the test videos should be ~0.75 for SUNS batch and ~0.67 for SUNS online. The processing time depends on the hardware. When executing on a laptop (Intel Core i5-6200U dual-core CPU, NVIDIA 940MX GPU), the training takes ~5.2 hours in total, testing SUNS batch takes ~6 seconds per video, and testing SUNS online takes ~27 seconds per video. When executing on a desktop computer (AMD 1920X 12-core CPU, NVIDIA Titan RTX GPU), the training takes ~50 minutes in total, testing SUNS batch takes ~1.5 seconds per video, and testing SUNS online takes ~20 seconds per video. When executing on another desktop computer (Intel i7-6800K 6-core CPU, NVIDIA GTX 1080 GPU), the training takes ~90 minutes in total, testing SUNS batch takes ~2 seconds per video, and testing SUNS online takes ~18 seconds per video.

Alternatively, you can also run `demo_pipeline_1to3.bat` instead of `demo_pipeline.bat`. The pipeline `demo_pipeline.bat` does standard leave-one-out cross validation on the four example videos. The pipeline `demo_pipeline_1to3.bat` trains the CNN model and post-processing parameters on one video, and tests on the remaining videos. 

Expected average F1 score
|	|Train	|Batch	|Online	|Track|
|:------:|:------:|:------:|:------:|:------:|
|train 3 test 1	|0.77	|0.73	|0.66	|0.68|
|train 1 test 3	|0.79	|0.73	|0.64	|0.66|

Example running time
|CPU	|GPU	|Train 3 to 1<br>(total)	|Train 1 to 3<br>(total)	|Batch<br>(average)	|Online<br>average)	|Track<br>(average)|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|AMD 1920X<br>12-core	|NVIDIA Titan RTX|	|	|1.5 s	|20 s	|	|
|Intel i7-6800K<br>6-core	|NVIDIA GTX 1080|	|	|2.0 s	|18 s	|	|
|Intel i5-6200U<br>dual-core	|NVIDIA 940MX	|5.4 h	|1.9 h	|6.0 s	|35 s	|36 s|


# Links to Datasets and Manual Markings
In our paper, we used two-photon imaging videos from [Allen Brain Observatory dataset](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS), [Neurofinder Challenge website](https://github.com/codeneuro/neurofinder), and [CaImAn dataset](https://github.com/flatironinstitute/CaImAn). We used the manual markings of Allen Brain Observatory and Neurofinder from [STNeuroNet](https://github.com/soltanianzadeh/STNeuroNet), and used the manual markings of CaImAn dataset from [CaImAn dataset](https://github.com/flatironinstitute/CaImAn). A more detailed instruction is given under the folder `paper reproduction`. 


# Citing 
If you use any part of this software in your work, please cite Bao et al. 2020:


# Licensing and Copyright
SUNS is released under [the GNU License, Version 2.0](LICENSE).


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
