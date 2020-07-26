# SUNS

Shallow UNet Neuron Segmentation (SUNS) is an automatic algorithm to segment active neurons from two-photon calcium imaging videos. It used temporal filtering and whitening schemes to extract temporal features associated with active neurons, and used a compact shallow U-Net to extract spatial features of neurons.


### System Requirements
* Anaconda with Python 3.7
* Tensorflow-gpu 1.15 (CUDA Toolkit 10.0 and cuDNN v7.6.5 required. Detailed instructions can be found [here][cuda-link].)

[cuda-link]: https://www.tensorflow.org/install/gpu

### Documentation
The how-to guides are available on [the Wiki][wiki-link].

[wiki-link]: https://github.com/YijunBao/Shallow-UNet-Neuron-Segmentation_SUNS/wiki

### Installation

* Install Anaconda with Python 3.7
* Launch Anaconda prompt and type the following in order (pathoffile is the directory to which the provided files were downloaded to):
```bash
cd pathoffile
cd installation
conda env create -f environment_suns_2.yml -n suns
```
* Go to Folder “<Anaconda root>/envs/suns/Lib/site-packages/fissa”, overwrite “core.py” with the one provided in the “installation” folder. If the dataset you used is less than 4 GB, you can skip this step. 


#### Link to Datasets:

[Allen Brain Observatory dataset][Allen-github]

[Neurofinder Challenge website][nf-website]

[CaImAn dataset][CaImAn-github]


[Allen-github]: https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS
[CaImAn-github]: https://github.com/flatironinstitute/CaImAn
[nf-website]: https://github.com/codeneuro/neurofinder

### Citing 

If you use any part of this software in your work, please cite Bao et al. 2020:


### Licensing and Copyright

SUNS is released under [the GNU License, Version 2.0](https://github.com/soltanianzadeh/STNeuroNet/LICENSE).


