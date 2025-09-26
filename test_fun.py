#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:56:21 2024

"""

import torch
import cornucopia as cc
import numpy as np
from torch.utils.data import Dataset
import time
from torch import nn
import nibabel as nib
import sys


#part of this code was written by Etienne Chollet (https://github.com/EtienneChollet/oct_vesselseg.git)


def test_fun(dataloader,model,DEVICE='cuda',normalize=True, clip=False):    
#     """
#     A testing function
#     """
    
     model.eval()     
     
     num_batches = len(dataloader.dataset)
     #print(num_batches)
     
     rescale = cc.QuantileTransform()
     #print(dataloader.dataset[0].shape)
     pred_all = np.zeros([num_batches,dataloader.dataset[0].shape[1],dataloader.dataset[0].shape[2],dataloader.dataset[0].shape[3]])
     #print(pred_all.shape)
     model.to(DEVICE)
# 
#     # Disable gradient computation and reduce memory consumption.
     with torch.no_grad():
         for n_batch, image in enumerate(dataloader):
             
             image = image.to(DEVICE).float()
             if normalize:
                 image = rescale(image)
             if clip:
                torch.clamp(image,min=0,max=1,out=image)            
             test_pred = model(image)
             pred_all[n_batch,:,:,:] = test_pred.detach().cpu().squeeze().numpy()
             
     return pred_all        
          
            

           
class RealVolume(object):
    """
    Base class for real volumetric data. ETienne's code
    """
    def __init__(self,
                 input:{torch.Tensor, str},
                 mask:{torch.Tensor, str}=None,
                 patch_size:int=256,
                 step_size:int=256,
                 binarize:bool=False,
                 binary_threshold:int=0.5,
                 normalize:bool=False,
                 pad_it:bool=False,
                 padding_method:str='reflect', # change to "reflect"
                 device:str='cuda',
                 dtype:torch.dtype=torch.float32,
                 patch_coords_:bool=False,
                 trainee=None
                 ):

        """
        Parameters
        ----------
        input : {torch.Tensor, 'path'}
            Tensor of entire tensor or path to nifti.
        mask : {None, torch.Tensor, 'path'}
        patch_size : int
            Size of patch with which to partition tensor into.
        step_size : int {256, 128, 64, 32, 16}
            Size of step between adjacent patch origin.
        binarize : bool
            Whether to binarize tensor.
        binary_threshold : float
            Threshold at which to binarize (must be used with binarized=True)
        normalize: bool
            Whether to normalize tensor.
        pad_ : bool
            If tensor should be padded.
        padding_method: {'replicate', 'reflect', 'constant'}
            How to pad tensor.
        device: {'cuda', 'cpu'}
            Device to load tensor onto.
        dtype: torch.dtype
            Data type to load tensor as.

        Attributes
        ----------
        volume_nifti
            Nifti represnetation of volumetric data.

        Notes
        -----
        1. Normalize
        2. Binarize
        3. Convert to dtype
        """
        self.input=input
        self.dtype=dtype
        self.patch_size=patch_size
        self.step_size=step_size
        self.binarize=binarize
        self.binary_threshold=binary_threshold
        self.normalize=normalize
        self.device=device
        self.pad_it=pad_it
        self.padding_method=padding_method
        self.tensor, self.nifti, self.affine = self.load_tensor(
            self.input,
            normalize=self.normalize,
            pad_it=self.pad_it
            )
        self.shape = self.tensor.shape
        #self.mask_tensor, self.mask_nifti, self.mask_affine = self.load_tensor(mask)
        
    def load_tensor(
            self,
            input,
            name:str='tensor',
            normalize:bool=False,
            pad_it:bool=False,
            binarize:bool=False
            ):
        """
        Prepare volume.

        Steps
        -----
        1. Load input volume if given path
        2. Convert to float32
        3. Detach from graph
        """
        if isinstance(input, str):
            # Getting name of volume (will be used later for saving prediction)
            #self.tensor_name = input.split('/')[-1].strip('.nii')
            # Get directory location of volume (will also use later for saving)
            #self.volume_dir = self.input.strip('.nii').strip(self.tensor_name).strip('/')
            nifti = nib.load(input)
            # Load tensor on device with dtype. Detach from graph.
            tensor = nifti.get_fdata()
            affine = nifti.affine
        elif isinstance(input, torch.Tensor):
            tensor = input.to(self.device).to(self.dtype).detach()
            nifti = None
        #volume_info(tensor, 'Raw')
        if normalize == True:
            tensor = self.normalize_volume(tensor)
        # Needs to be a tensor for padding operations
        tensor = torch.as_tensor(tensor, device=self.device).detach()
        if pad_it == True:
            tensor = self.pad_volume(tensor)
        if self.binarize == True:
            tensor[tensor <= self.binary_threshold] = 0
            tensor[tensor > self.binary_threshold] = 1
        tensor = tensor.to(self.dtype)
        return tensor, nifti, affine


    def normalize_volume(self, input):
        print('\nNormalizing volume...')
        input = torch.from_numpy(input).to(self.device)
        input -= input.min()
        input /= input.max()
        #input = QuantileTransform(pmin=0.02, pmax=0.98)(input)
        return input
        #volume_info(self.tensor, 'Normalized')


    def pad_volume(self, tensor):
        """
        Pad all dimensions of 3 dimensional tensor and update volume.
        """
        print('\nPadding volume...')
        # Input tensor must be 4 dimensional [1, n, n, n] to do padding
        padding = [self.patch_size] * 6 # Create 6 ele list of patch size
        tensor = torch.nn.functional.pad(
            input=tensor.unsqueeze(0),
            pad=padding,
            mode=self.padding_method
        )[0]
        #volume_info(tensor, 'Padded')
        return tensor
    
    def make_mask(self, n_clusters=3):
        backend = dict(dtype=self.tensor.dtype, device=self.tensor.device)
        
        preprocessed_vol = skimage.filters.gaussian(self.tensor, 10)
        preprocessed_vol = preprocessed_vol.reshape(-1, 1)
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
        means = kmeans.fit(preprocessed_vol)
        segmented_vol = kmeans.cluster_centers_[kmeans.labels_]

        segmented_vol = torch.from_numpy(segmented_vol.reshape(self.shape))
        labelmask = torch.from_numpy(kmeans.labels_.reshape(self.shape))

        means = torch.unique(segmented_vol)
        labels = torch.unique(labelmask)

        for i in range(len(means)):
            segmented_vol[segmented_vol == means[i]] = labels[i]
        
        return segmented_vol.to(torch.int16).numpy()



class predictSingleImage(RealVolume,Dataset):    
#     """
#     A testing function
#     """

    def __init__(self, volume_path, model, DEVICE='cuda', normalize_patches=True, normalize_image=False, clip_input_patch=False, **kwargs):
        super().__init__(volume_path, **kwargs)
        #here I run the init method of RealVolume with the **kwargs, which 
        #creates self.tensor, and self.affine *among other things)

        self.model = model
        self.device = DEVICE
        self.normalize_patches = normalize_patches
        self.normalize_image = normalize_image
        self.clip = clip_input_patch
        self.rescale = cc.QuantileTransform()
        #self.input_image = input_image
        # self.activation = _make_activation(activation)


        
        min_scale = 1/2
        half_filter_1d = torch.linspace(min_scale, torch.pi/2, self.patch_size//2).sin().to(self.device)
        filter_1d = torch.concat([half_filter_1d, half_filter_1d.flip(0)])
        self.patch_weight = filter_1d[:, None, None] * filter_1d[None, :, None] * filter_1d[None, None, :]

        self.model.eval()
        self.model.to(self.device)   

        self.patch_coords() # calls method patch_coords ehich fills self.complete_patch_coords with all the patch coordinates


    def __len__(self):
        return len(self.complete_patch_coords)
    
    def __getitem__(self, idx):
        working_patch_coords = self.complete_patch_coords[idx]
        # Generating slices for easy handling
        x_slice = slice(*working_patch_coords[0])
        y_slice = slice(*working_patch_coords[1])
        z_slice = slice(*working_patch_coords[2])        
        patch = self.tensor[x_slice, y_slice, z_slice].detach()
        coords = [x_slice, y_slice, z_slice]
        return patch, coords

    # def _make_activation(activation):
    #     if isinstance(activation, str):
    #         activation = getattr(nn, activation)
    #     activation = (activation() if inspect.isclass(activation)
    #                 else activation if callable(activation)
    #                 else None)
    #     return activation 

    
    def patch_coords(self):
        self.complete_patch_coords = []
        tensor_shape = self.tensor.shape
        # used to be x_coords = [[x, x+self.patch_size] for x in range(0, tensor_shape[0] - self.step_size + 1, self.step_size)]
        x_coords = [[x, x+self.patch_size] for x in range(self.step_size, tensor_shape[0] - self.patch_size, self.step_size)]
        y_coords = [[y, y+self.patch_size] for y in range(self.step_size, tensor_shape[1] - self.patch_size, self.step_size)]
        z_coords = [[z, z+self.patch_size] for z in range(self.step_size, tensor_shape[2] - self.patch_size, self.step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    self.complete_patch_coords.append([x, y, z])


    def predictSinglePatch(self, idx:int):

        patch, coords = self.__getitem__(idx)
        patch = patch.to(self.device).float()
        #print(patch.shape, patch.device)
        patch = patch.unsqueeze(0).unsqueeze(0) #add batch and channel dimensions
        #print(patch[0,0,0:10,0:10,0:10])
        if self.normalize_patches == True and patch.sum() != 0:
            patch = self.rescale(patch)
        if self.clip:
            torch.clamp(patch, min=0, max=1, out=patch)

        prediction_patch = self.model(patch).squeeze() #activation is already included in the model
        # prediction_patch = self.activation(prediction_patch).squeeze()

        
        self.prediction[coords[0], coords[1], coords[2]] += (prediction_patch * self.patch_weight) #imprint_tensor is big volume that we initialize as empty. 
        #then we sum all the predictions in there (weighted by the patch weights to give more importance to the center)
    

    def predict(self):
        # if input_image.dtype != torch.float32:
        #     input_image = input_image.to(torch.float32)
        #     print('Input tensor needs to be float32! Converting it now...')
        n_patches = len(self)
        if self.normalize_image == True:
            self.tensor = self.rescale(self.tensor)
        t0 = time.time()
        self.prediction = torch.zeros_like(self.tensor)
        print('Starting predictions!')
        with torch.no_grad():            
            for i in range(n_patches):
                self.predictSinglePatch(i) #it updates self.prediction
                if (i+1) % 10 == 0:
                    total_elapsed_time = time.time() - t0
                    average_time_per_pred = round(total_elapsed_time / (i+1), 3)
                    sys.stdout.write(f"\rPrediction {i + 1}/{n_patches} | {average_time_per_pred} sec/pred | {round(average_time_per_pred * n_patches / 60, 2)} min total pred time")
                    sys.stdout.flush()

        # Remove padding
        s = slice(self.patch_size, -self.patch_size)
        self.prediction = self.prediction[s, s, s]
        redundancy = ((self.patch_size ** 3) // (self.step_size ** 3))

        print(f"\n\n{redundancy}x Averaging...")
        self.prediction /= redundancy
        self.prediction = self.prediction.cpu().numpy()
        del self.tensor #free memory
        return self.prediction, self.affine #return the prediction and the affine matrix of the input image
    





class test_convolve(nn.Module):   


    def __init__(self, volume_path, model,patch_size, step_size, DEVICE='cuda',normalize_patches=True, normalize_image = False, clip_input_patch=False):
        super().__init__()
     
        self.model = model
        self.device = DEVICE
        self.normalize_patches = normalize_patches
        self.normalize_image = normalize_image
        self.clip = clip_input_patch
        self.rescale = cc.QuantileTransform()
        self.patch_size = patch_size
        self.step_size = step_size
        self.volume_path = volume_path
       
        self.predictClass = predictSingleImage( 
                self.volume_path,
                self.model,
                DEVICE=self.device,
                normalize_patches=self.normalize_patches, 
                normalize_image=self.normalize_image,
                clip_input_patch=self.clip,
                dtype=torch.float32,
                patch_size=self.patch_size,
                step_size=self.step_size,
                pad_it=True,
                padding_method='reflect'
                        )

    def forward(self):

       
        
       
    # 
    #     # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            

            

    #this creates stuff for the single image, so it needs to be initialized here inside the loop
            test_pred, affine = self.predictClass.predict() #calls the predict method of the RealPredict class, which runs the prediction on all patches

            test_pred = test_pred.squeeze()

        return test_pred, affine

