# SUNS

Shallow UNet Neuron Segmentation (SUNS) is an automatic algorithm to segment active neurons from two-photon calcium imaging videos. It used temporal filtering and whitening schemes to extract temporal features associated with active neurons, and used a compact shallow U-Net to extract spatial features of neurons.


### Documentation
The how-to guides are available on [the Wiki][wiki-link].

[wiki-link]: https://github.com/YijunBao/Shallow-UNet-Neuron-Segmentation_SUNS/wiki

### System requirement
* Operation system: Windows 10.
* Memory: ~6x file size of the raw video if the the raw video is in uint16 format. ~3x file size of the raw video if the the raw video is in float32 format. 
* A CUDA compatible GPU is preferred.

### Installation on Windows
* Install [Anaconda][Anaconda] with Python 3.7
* Launch Anaconda prompt and type the following in order (SUNS_python_root_path is the directory to which the provided files were downloaded to, such as `C:/Users/(username)/Documents/GitHub/Shallow-UNet-Neuron-Segmentation_SUNS`):
```bat
cd SUNS_python_root_path
cd installation
conda env create -f environment_suns_2.yml -n suns
```
* Go to the Anaconda environment foler, (such as `C:/ProgramData/Anaconda3/envs` or `C:/Users/(username)/.conda/envs`), and then go to folder `suns/Lib/site-packages/fissa`, overwrite `core.py` and `neuropil.py` with the files provided in the `installation` folder. If the dataset you used is less than 4 GB after the data type is converted to float32 and you don't mind a lot of text output from FISSA, you can skip this step. 

The installation should take less than half an hour in total. The first run of the software may take some additional time (up to 20 minutes on a laptop) to add the GPU, but this extra time will not occur in later runs.

[Anaconda]: https://www.anaconda.com/

### Demo
We provided a demo for all users to get familiar with our software. We provided four two-photon imaging videos as well as their manually marked neurons in `demo/data`. The demo will perform a cross validation over the four videos: train the CNN and search for optimal hyper-parameters using three videos, and test SUNS with the training output on the remaining video. 

To run the demo, launch Anaconda prompt and type the following script 
```bat
cd SUNS_python_root_path
cd demo
conda activate suns
demo_pipeline.bat
```
The demo contains three parts: training CNN and hyper-parameters, testing SUNS batch, and testing SUNS online. The output masks of SUNS batch will be in `demo/complete/output_masks`, and the output masks of SUNS batch will be in `demo/complete/output_masks online`. The average F1 score of the training videos should be ~0.8, and the average F1 of the test videos should be ~0.75 for SUNS batch and ~0.67 for SUNS online. The processing time depends on the hardware. When executing on a laptop (Intel Core i5-6200U dual-core CPU, NVIDIA 940MX GPU), the training takes ~5.2 hours in total, testing SUNS batch takes ~6 seconds per video, and testing SUNS online takes ~27 seconds per video. When executing on a desktop computer (AMD 1920X 12-core CPU, NVIDIA Titan RTX GPU), the training takes ~50 minutes in total, testing SUNS batch takes ~1.5 seconds per video, and testing SUNS online takes ~20 seconds per video. When executing on another desktop computer (Intel i7-6800K 6-core CPU, NVIDIA GTX 1080 GPU), the training takes ~90 minutes in total, testing SUNS batch takes ~2 seconds per video, and testing SUNS online takes ~18 seconds per video.


### Links to Datasets and Manual Markings:

In our paper, we used two-photon imaging videos from [Allen Brain Observatory dataset][Allen-github], [Neurofinder Challenge website][Neurofinder-website], and [CaImAn dataset][CaImAn-github]. We used the manual markings of Allen Brain Observatory and Neurofinder from [STNeuroNet][STNeuroNet-github], and used the manual markings of CaImAn dataset from [CaImAn dataset][CaImAn-github].

[Allen-github]: https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS
[CaImAn-github]: https://github.com/flatironinstitute/CaImAn
[Neurofinder-website]: https://github.com/codeneuro/neurofinder
[STNeuroNet-github]: https://github.com/soltanianzadeh/STNeuroNet

### Citing 

If you use any part of this software in your work, please cite Bao et al. 2020:


### Licensing and Copyright

SUNS is released under [the GNU License, Version 2.0](https://github.com/soltanianzadeh/STNeuroNet/LICENSE).


