# Vessynth - A flexible vessel segmentation method

## Installation

## Usage

After installing and activating the python environment, you can use the method with:

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
