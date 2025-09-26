# Vessynth - A flexible vessel segmentation method

## Installation

## Usage

```
python vessynth_test.py -i <vol> -o <outputDir> -mod <modality> [-th <threshold> -m <mask_vol>]
```

where
- ```<vol>``` is input nifti volume to segment. You can also provide more than one volume, e.g. -i vol1 vol2 vol3
- ```<outputDir>``` is output directory where segmentations are saved
- ```<modality>``` indicates the modality/contrast of the input volume. Accepted values are 'T2star' (for exvivo MRI for instance), 'OCT' (for Optical Coherence Tomography) and 'TOF' (for in vivo Time-Of-Flight MRA)

optional arguments are 
- ```<threshold>``` value to threshold the probablity to obtain a hard segmentation. default is 0.3
- ```<mask_vol>``` is a binary mask to apply to segmentation (e.g. 1 inside brain, 0 outside). Useful to remove noise outside brain
