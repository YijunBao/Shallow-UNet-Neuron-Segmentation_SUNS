- [Get raw videos and GT masks from public dataset](#get-raw-videos-and-gt-masks-from-public-dataset)
  - [Allen Brain Observatory (ABO) dataset](#allen-brain-observatory-abo-dataset)
  - [Neurofinder Challenge dataset](#neurofinder-challenge-dataset)
  - [CaImAn dataset](#caiman-dataset)
- [Convert manual labels into sparse matrices](#convert-manual-labels-into-sparse-matrices)
- [Generate convolution kernel for temporal filtering](#generate-convolution-kernel-for-temporal-filtering)
- [Reproduce the results in our paper](#reproduce-the-results-in-our-paper)


# Get raw videos and GT masks from public dataset
In our paper, we used two-photon imaging videos from Allen Brain Observatory dataset, Neurofinder Challenge website, and CaImAn dataset. We used the manual markings of Allen Brain Observatory and Neurofinder from the STNeuroNet repository, and used the manual markings of CaImAn dataset from CaImAn dataset. A more detailed instruction is given below.


## Allen Brain Observatory (ABO) dataset
The ABO dataset is available in [Allen Institute](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS). You may need a Amazon AWS account to download them. We used ten videos from 275 um layer, {'524691284', '531006860', '502608215', '503109347', '501484643', '501574836', '501729039', '539670003', '510214538', '527048992'}, and 10 videos from 175 um layer, {'501271265', '501704220', '501836392', '502115959', '502205092', '504637623', '510514474', '510517131', '540684467', '545446482'}. We used the manual labels of [275 um layer](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO/Layer275/FinalGT) and [175 um layer](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO/Layer175/FinalGT) created by Soltanian-Zadeh et al. We also used the code [create_h5_video_ABO.m](utils/create_h5_video_ABO.m) modified from the same STNeuroNet repository to crop each video to the first 20% durations and the center parts, so that the video sizes are changed from 512 x 512 x ~115,000 to 487 x 487 x ~23,000. Set the folders correctly, and run the code twice by setting `layer = 275` and `layer = 175`. 


## Neurofinder Challenge dataset
The Neurofinder dataset is available in [Neurofinder](https://github.com/codeneuro/neurofinder). We used 6 training videos, {01.00, 01.01, 02.00, 02.01, 04.00, 04.01}, and 6 testing videos, {01.00.test, 01.01.test, 02.00.test, 02.01.test, 04.00.test, 04.01.test}. We used the manual labels of [training video](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/Neurofinder/train/Grader1) and [testing video](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/Neurofinder/test/Grader1) created by Soltanian-Zadeh et al. We also used the code [create_h5_video_NF.m](utils/create_h5_video_NF.m) modified from the same STNeuroNet repository to crop the center parts. Set the folders correctly, and run the code twice by setting `opt.type = 'train'` and `opt.type = 'test'`. Rename the GT mask files by replacing the short video number with complete video number (e.g. replacing "100" with "01.00" or "01.00.test"). 


## CaImAn dataset
In the following procedures, we use MATLAB to convert the raw video from series of ".tif" or ".tiff" images to ".h5" files and convert the ground truth labels (GT masks) from ".json" files to ".mat" files. The code requires a MATLAB package [JSONLab](https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files), so download it first and put it in a path that MATLAB can access. 

The video and manual labels of the CaImAn dataset is available [here](https://zenodo.org/record/1659149). The video can also be downloaded [here](https://users.flatironinstitute.org/~neuro/caiman_paper). We used the videos J115, J123, K53, and YST. The GT labels are provided in `WEBSITE_basic`. Download the five ".zip" files and put them in the same folder, then unzip them. Run [utils/video_masks_CaImAn.m](utils/video_masks_CaImAn.m) to generate the ".h5" files for raw videos and the ".mat" files for GT masks. Each video is divided into a set of 4 quarter-sized sub-videos with equal size and similar numbers of neurons. 


# Convert manual labels into sparse matrices
All the manual labels are now stored in the form of a 3D array. For convenience of use, we convert them to 2D sparse matrices by running [utils/generate_sparse_GT.m](utils/generate_sparse_GT.m)


# Generate convolution kernel for temporal filtering
We generated the temporal filter kernel by averaging fluorescence responses of the GT neurons to calcium transients with moderate peak SNR between 5 and 8 aligned to their peaks. The generated filter kernels are provided in `paper reproduction/Generalization_test` as ".h5" files. Alternatively, you can also use the code under `utils/temporal filter` to regenerate them. After setting the folders, run `calculate_traces_bgtraces_ABO.m` and `temporal_filter_ABO.m` to obtain the filter kernel for the ABO dataset. Run other similar functions to obtain the filter kernels for Neurofinder (NF) and CaImAn (CM) datasets. 


# Reproduce the results in our paper
We used the conda environment exported as `installation\environment_suns_test.yml` to run all of our tests. The major results in our paper can be reproduced by running the `.bat` files under the sub folders `ABO`, `Neurofinder`, and `CaImAn dataset`. Some module versions are older than the latest versions installed from `installation\environment_suns.yml`. We showed all of our trained CNNs and optimized postprocessing hyperparameters in `training results`. We showed all of our output masks, together with the output masks of the peer algorithms, in `output masks all methods`. The results of SUNS were saved in Python using C order, while the results of the other methods were saved in MATLAB using F order, so a permutation/transpose is required to align their dimensions (i.e., `permute(Masks,[3,2,1])` in MATLAB, or `np.transpose(Masks,[2,1,0])` in Python). 