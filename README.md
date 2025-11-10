# Vessynth - A flexible vessel segmentation method

## Installation

1. Clone this repo and set it as your current working directory
``` 
git clone https://github.com/chiara-mauri/Vessynth.git
cd Vessynth
```
2. Create and activate a conda environment:

- Option 1: Use the provided yaml file:

```
   conda env create -f vessynth-env.yml
   conda activate vessynth-env
```

- Option 2: Create the environment and manually install the required packages:
```
   conda create -n vessynth-env python=3.10
   conda activate vessynth-env
   pip install cornucopia
   pip install pandas==2.3.3 tensorstore==0.1.78 boto3==1.40.69
```


## Download the models 

Download the folder 'weights' from the Dandiset 1062/models_vessel_seg/
https://dandiarchive.org/dandiset/001602/draft/files?location=models_vessel_seg&page=1 and copy it into the 'models' folder of this repo


## Usage

Now you can use the method with:

```
python path/to/repo/vessynth_test.py -i <vol> -o <outputDir> -mod <modality> [-th <threshold> -m <mask_vol>]
```

where the required arguments are:
- ```<vol>``` is input nifti volume to segment. 
- ```<outputDir>``` is output directory where segmentations are saved
- ```<modality>``` indicates the modality/contrast of the input volume. Accepted values are
   - 'T2star': for exvivo MRI and all T2star-based contrasts. Vessels are both bright and dark. Mesoscopic resolution (100-200um)
   - 'HipCT': for Hierarchical Phase-Contrast Tomography. Dark vessels. Resolution ~ 30um
   -  'OCT': for Optical Coherence Tomography. Dark vessels. Resolution ~ 20um
   -  'TOF': for in vivo Time-Of-Flight Magnetic Resonance angiography. Bright vessels. Resolution ~ tentatively from 160um iso to 500um x 500um x 800um

optional arguments are:
- ```<threshold>``` value used to threshold the 'vessel probablity' to obtain a hard segmentation. default is 0.3
- ```<mask_vol>``` a binary mask applied to the segmentation (e.g. 1 inside brain, 0 outside). Useful to remove noise outside brain
