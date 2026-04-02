# VesSynth - A Robust Cross-Scale Cross-Modal 3D Vessel Segmentation Method

## Installation

1. Clone this repo and set it as your current working directory
``` 
git clone https://github.com/chiara-mauri/Vessynth.git
cd Vessynth
```
2. Create and activate a conda environment:

- Option 1: Use the provided yaml file:

```
   conda env create -f vessynth-env.yml (or for Mac: conda env create -f vessynth-env-macOS.yml)
   conda activate vessynth-env
```

- Option 2: Create the environment and manually install the required packages:
```
   conda create -n vessynth-env python=3.10
   conda activate vessynth-env
   pip install cornucopia
   pip install pandas==2.3.3 tensorstore==0.1.78 boto3
```


## Download the models 

Download the 'models' folder from https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/vesSynth/ 
unzip it and copy it in this repo
<!-- from the Dandiset 1062/models_vessel_seg/
https://dandiarchive.org/dandiset/001602/draft/files?location=models_vessel_seg&page=1 --> 



## Usage

Now you can use the method with:

```
python path/to/repo/vessynth_test.py -i <vol> -o <outputDir> -mod <modality> [-th <threshold> -m <mask_vol> -c <cutout> -nw]
```

where the required arguments are:
- ```<vol>``` is input nifti volume to segment. 
- ```<outputDir>``` is output directory where segmentations are saved
- ```<modality>``` indicates the modality/contrast of the input volume. Accepted values are
   - 'T2star': for exvivo MRI and all T2star-based contrasts. Vessels are both bright and dark. Mesoscopic resolution (100-300um)
   - 'HiPCT': for Hierarchical Phase-Contrast Tomography. Dark vessels. Resolution ~ 20-30um
   -  'OCT': for Optical Coherence Tomography. Dark vessels. Resolution ~ 20um
   -  'TOF': for in vivo Time-Of-Flight Magnetic Resonance angiography. Bright vessels. Flexible resolution, from ~150um iso to 500um x 500um x 1mm
   -  'fibers': for bright fiber bundles/axons across modalities (experimental)

optional arguments are:
- ```<threshold>``` value used to threshold the 'vessel probablity' to obtain a hard segmentation. default is 0.3. Value reccommended for TOF = 0.5
- ```<mask_vol>``` a binary mask applied to the segmentation (e.g. 1 inside brain, 0 outside). Useful to remove noise outside brain
- ```<cutout>``` a bounding box to identify ROI (```-zc x1 x2 y1 y2 z1 z2```)
- ```-nw```, ```--no_weights``` do NOT use Gaussian weights when computing segmentation on a patch
