# Deep Learning in Brain MRI: Effect of Data Leakage Due to Slice-level Split Using 2D Convolutional Neural Networks

This repository contains the libraries and code required to reproduce the study in [1].

Briefly, we illustrated the effect of data leakage caused by inappropriate train/validation split of 3D MR image data. Specifically, we trained three different 2D convolutional neural network (CNN) architectures using correct (subject-level) and incorrect (slice-level) data split on public and private datasets. The difference in the model's performance trained following both incorrect and incorrect data split is demonstrated to emphasize the extent of the over-estimation of the model's performance caused by data leakage.



## Datasets
Since we cannot redistribute MR data, the user is expected to get T1-weighted MR images from the following open access datasets (in NIFTI format).

- Open Access Series of Imaging Studies (OASIS), available at (https://www.oasis-brains.org/)    (We considered the same subjects used in [4])

- Alzheimer Disease Neuroimaging Initiative (ADNI), accessible at (https://adni.loni.usc.edu/). Information on how to download the data can be found from [ADNI_README.md](https://github.com/Imaging-AI-for-Health-virtual-lab/Slice-Level-Data-Leakage/blob/main/docs/ADNI_README.md).

- Parkinson's Progression Markers Initiative (PPMI), found from (https://www.ppmi-info.org/). Steps to download the data are listed [PPMI_README.md](https://github.com/Imaging-AI-for-Health-virtual-lab/Slice-Level-Data-Leakage/blob/main/PPMI_README.md).

To let other researchers reproduce our results, we provided in the supplementary material of [our preprint](https://doi.org/10.21203/rs.3.rs-464091/v1) the specific subject identification numbers (IDs) we have selected to create our datasets [1].

## 3D MRI data pre-processing
The T1-weighted images required a pre-processing, including a co-registration step with a standard space and a skull-stripping procedure.

We show an example of pre-processing, by using [ANTS](http://stnava.github.io/ANTs/) [2] and [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) [3] scripts on the following T1-weighted images of two AD patients of ADNI dataset (https://adni.loni.usc.edu/):

- ADNI_002_S_5018_MR_MPRAGE_br_raw_20121112145218294_127_S174291_I346242.nii.gz

- ADNI_006_S_4153_MR_MPRAGE_br_raw_20111107134001421_166_S128231_I264989.nii.gz





1. Registration with standard template:

```
ants.sh 3 $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz ADNI_002_S_5018_MR_MPRAGE_br_raw_20121112145218294_127_S174291_I346242.nii.gz

ants.sh 3 $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz ADNI_006_S_4153_MR_MPRAGE_br_raw_20111107134001421_166_S128231_I264989.nii.gz

```



The outputs should be:
- ADNI_002_S_5018_MR_MPRAGE_br_raw_20121112145218294_127_S174291_I346242deformed.nii.gz

- ADNI_006_S_4153_MR_MPRAGE_br_raw_20111107134001421_166_S128231_I264989deformed.nii.gz



2. Concatenate the registered images in a unique 4D volume:

```
fslmerge -t ADNI_example.nii.gz ADNI_002_S_5018_MR_MPRAGE_br_raw_20121112145218294_127_S174291_I346242deformed.nii.gz ADNI_006_S_4153_MR_MPRAGE_br_raw_20111107134001421_166_S128231_I264989deformed.nii.gz

```



3. Mask the 4D volume with the standard space brain mask:

```
fslmaths ADNI_example.nii.gz -mas $FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz ADNI_example_skullstripped.nii.gz
```



In the following documentation, 'ADNI_example_skullstripped.nii.gz' is the input 4D NIfTI volume, containing all subjects concatenated one to each other. The csv label file should be written consistent with this 4D NIfTI volume (in this guide the csv file 'ADNI_example_labels.csv' has been written consistent with the 4D NIfTI volume 'ADNI_example_skullstripped.nii.gz').



# Installation
## Clone Repository with Git
Clone the Slice-Level-Data-Leakage

```
git clone https://github.com/Imaging-AI-for-Health-virtual-lab/Slice-Level-Data-Leakage.git

cd Slice-Level-Data-Leakage

```
## Install Packages with Anaconda
Next, download and install [Anaconda](https://www.anaconda.com/products/individual). Create a new conda environment that includes all the dependencies for this project from a requirements_list.txt file.


```
conda create --name <new_env_name> --file requirements_list.txt

```

Activate the conda environment.

```
conda activate <new_env_name> 

```

Finally, install the remaining packages with pip.

```
pip install keras-vis==0.5.0 image-classifiers==0.1 tf-explain==0.3.0

```



##  Installation with Docker
First, pull the docker image by visiting the docker hub repository located at [Docker](https://hub.docker.com/repository/docker/ai4healthvlab/slice-level-data-leakage).

```
docker pull ai4healthvlab/slice-level-data-leakage

```




## Usage
Run the script _model_training.py_:

```
python3 model_training.py -h

usage: model_training.py [-h] [--save_path SAVE_PATH] nifti_path labels_path arch_config_path data_CV_config_path


positional arguments:

nifti_path:  4D NIfTI volume path
labels_path: csv labels file path
arch_config_path:  path of the json file containing the CNN architecture configuration
data_CV_config_path:  path of the json file containing the data selection and CV setting


optional arguments:

-h, --help  show this help message and exit
--save_path SAVE_PATH     location to save the results



Example:

python3 model_training.py ./data/ADNI_example_skullstripped.nii.gz ./data/ADNI_example_labels.csv ./config/arch_pars.json ./config/data_CV_pars.json --save_path path_to_save_results
```



- arch_config_path: json file containing configuration for CNN architecture. The parameters are:
1. "pretrained_model": the choice of network architecture ("vgg1", "vgg2", "resnet18").

2. "optimizer": optimizer options ("adam", "SGD").

3. "epochs": number of epochs to train the model.

4.  "batch_size": batch size during model training.




- data_CV_config_path: json file containing configuration for data selection and CV setting. The parameters include:

1. "dataset": string representing the name of the dataset.
2.  "data_size": option to use all the data ("all") or a sub-sample of the dataset ("subset").

3. "sample_size": integer number to choose number of subjects to use for a sub-sampled dataset. Necessary if "data_size" is set as "subset".

4. "Nslices": number of slices to select.

5. "slicing_plane": string to choose slicing plane ("axial", "sagittal" or "coronal").

6. "enable_hist_plot": boolean variable for plotting the histogram of the selected number of slices.

7. "data_resize": boolean variable for resizing 2D images or not.

8. "label": string column header of the label's file.

9. "cv_method": string to choose between cross validation ("cv") or nested cross validation ("grid_search").

10. "Nfolds": number of cross validation folds. Required if "cv_method" is set as "cv".

11. "Ninner": integer representing number of inner folds. Required if "cv_method" is set as "grid_search".

12. "Nouter": integer representing number of outer folds. Required if "cv_method" is set as "grid_search".

13. "cv_split": string to choose data split method ("slice-level" and "subject-level")




## Usage with Docker


```
docker run -v "arguments_path:docker_image_path" --rm --name doc_container doc_image docker_image_path/nifty_file docker_image_path/labels_file docker_image_path/arch_config_file docker_image_path/data_config_file


arguments_path         path to argument files
docker_image_path      path to a location where the docker image is loaded
doc_container          name of dcoker container
doc_image              name of docker image
nifty_file             nifty fille name
labels_file            labels file name
arch_config_file       model architecture configuration file
data_config_file       data configuration file name



Example

docker run -v "Path_to_arguments:Path_to_docker_image" --rm --name Name_doc_container Name_docker_image Path_to_docker_image/ADNI_example_skullstripped.nii.gz Path_to_docker_image/ADNI_example_labels.csv Path_to_docker_image/arch_pars.json Path_to_docker_image/data_pars.json 

```


**References**



[1] Yagis, E., Atnafu, S.W., de Herrera, A.G.S., Marzi, C., Giannelli, M., Tessa, C., Citi, L. and Diciotti, S., 2021. Deep Learning in Brain MRI: Effect of Data Leakage Due to Slice-level Split Using 2D Convolutional Neural Networks. https://doi.org/10.21203/rs.3.rs-464091/v1.



[2] B. B. Avants et al., 2011. A reproducible evaluation of ANTs similarity metric performance in brain image registration.  _Neuroimage_, Volume 54(3), pp. 2033-2044.



[3] M. Jenkinson, C.F. Beckmann, T.E. Behrens, M.W. Woolrich, S.M. Smith. FSL. _NeuroImage_, 62:782-90, 2012


[4] M. Hon and N.M. Khan, 2017, November. Towards Alzheimer's disease classification through transfer learning. In 2017 IEEE International conference on bioinformatics and biomedicine (BIBM) (pp. 1166-1169). 
