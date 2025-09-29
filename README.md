# Vessynth - A flexible vessel segmentation method

## Installation

First create a conda environment:

- Option 1: Use the provided yaml file:

```
   conda env create -f vessels-env-test.yml
   conda activate vessels-env-test.yml
```

- Option 2: Create the environment and install the dependencies:
```
   conda create -n vessels-env-test python=3.10
   conda activate vessels-env-test
   conda install pytorch=1.13
   pip install cornucopia
```

Then clone this repo
``` 
git clone https://github.com/chiara-mauri/Vessynth.git
```

## Download the models 

Download the folder 'weights' from the Dandiset 1062
https://dandiarchive.org/dandiset/001602/draft/files?location=models_vessel_seg&page=1 and copy it into the 'models' folder of this repo


## Usage

Now you can use the method with:

```
python vessynth_test.py -i <vol> -o <outputDir> -mod <modality> [-th <threshold> -m <mask_vol>]
```

where the required arguments are:
- ```<vol>``` is input nifti volume to segment. 
- ```<outputDir>``` is output directory where segmentations are saved
- ```<modality>``` indicates the modality/contrast of the input volume. Accepted values are 'T2star' (for exvivo MRI - bright and dark vessels), 'HipCT' (for Hierarchical Phase-Contrast Tomography - dark vessels) and 'TOF' (for in vivo Time-Of-Flight MRA - bright vessels)

optional arguments are:
- ```<threshold>``` value used to threshold the 'vessel probablity' to obtain a hard segmentation. default is 0.3
- ```<mask_vol>``` a binary mask applied to the segmentation (e.g. 1 inside brain, 0 outside). Useful to remove noise outside brain
