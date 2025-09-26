#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:14:03 2024
@author: cm1991
"""


import glob
import time
import torch
import argparse
import nibabel as nib
from test_fun import test_convolve
from utils.networks import SegNet
import os
import numpy as np
import json



if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Method for vessel segmentation.')

    parser.add_argument('-i', '--inpvol', type=str, nargs='+', required=True,
                        help='path to volumes you want to predict on. It can be a list of volumes.') 
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help='output directory to save predictions.')
    parser.add_argument('-m', '--modality', type=str, required=True,
                        help='Type of modality. Allowed: OCT, T2star, TOF.')
    parser.add_argument('-t', '--threshold', type=float, nargs='+', default=[0.3],
                        help='Threshold to apply to the predictions. Default is 0.3. It can be a list of thresholds.')
    parser.add_argument('--masks', type=str, nargs='+', default=None,
                        help='List of masks to apply to the predictions. If None, no mask is applied. \
                            Masks should be in the same order as the volumes. Default is None.')

   
    #parameters of volumes to predict on
    
    # parser.add_argument('--save-native-space', action='store_true',
    #                     help='Whether to also save the predictions in native space. Default is False.')
    # parser.add_argument('--patch-size', type=int, default=128,
    #                     help='size of UNet input (and size of sliding prediction patch). Default is 128.')
    # parser.add_argument('--step-size', type=int, default=32,
    #                     help='step size (in vx) between adjacent prediction patches. Default is 32.')

    
    
    args = parser.parse_args()
    
    
    volumes = args.inpvol
    outputdir = args.outdir
    modality = args.modality
    threshold = args.threshold
    mask_list = args.masks

    #hardcoded parameters
    patch_size = 128
    step_size = 32
    final_activation = 'Sigmoid'
    save_native_space = True

    if mask_list is not None:
        for i,mask_ in enumerate(mask_list):
            if mask_ == 'None':
               mask_list[i] = None
    
    print("-----------------------------------")
    print("Running vessel segmentation with Vessynth")
    print("-----------------------------------")

    print(f"\nVolumes to predict on: {volumes}")
    print(f"Output directory: {outputdir}")
    print(f"Modality: {modality}")
    print(f"Threshold: {threshold}")
    print(f"Mask list: {mask_list}")
    print(f"Predicting with patch size {patch_size} and step size {step_size}")


    model_path = './models/'
    
    if modality == 'OCT':
        model_to_load = glob.glob('./models/OCT_model*')[0]
    elif modality == 'T2star':
        model_to_load = glob.glob('./models/T2star_model*')[0]
    elif modality == 'TOF':
        model_to_load = glob.glob('./models/TOF_model*')[0]
    else:
        raise ValueError('Modality not recognized. Allowed: OCT, T2star, TOF.')
    

    if not os.path.exists(outputdir):
        print(f"Creating output directory: {outputdir}")
        os.makedirs(outputdir)


    t1 = time.time()
    with torch.no_grad():

        # Path to the JSON file containing the backbone dictionary
        json_path = os.path.join(model_path, f'segnet_model.json')
        # Read backbone_dict from the JSON file
        with open(json_path, 'r') as f:
            backbone_dict = json.load(f)
        print("\nLoaded model backbone info from", json_path)
        
        model = SegNet(ndim=3, in_channels=1, out_channels=1,
                        init_kernel_size=3, final_activation=final_activation, backbone='UNet', 
                        kwargs_backbone=backbone_dict)

        

        print('Loading model: ', model_to_load)
        #saved_model = torch.load(model_to_load, map_location=torch.device('cpu'))
        saved_model = torch.load(model_to_load) # for gpu
        
        if 'model_state_dict' in saved_model:
            model.load_state_dict(saved_model['model_state_dict'])
        elif 'model_state_dict_segnet' in saved_model:
            model.load_state_dict(saved_model['model_state_dict_segnet'])
        else:
            exception = f"Model state dict not found in {model_to_load}. Please check the file."
            raise Exception(exception)
        print('Model loaded successfully!')
        
        print(f"\nStarting predictions on {len(volumes)} volumes...")
        for vol_index, vol in enumerate(volumes):

            print(f"Processing volume {vol_index + 1}/{len(volumes)}: {vol}")

            prediction, affine = test_convolve(
                vol,
                model,
                patch_size,
                step_size,
                DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
                normalize_patches=True, 
                normalize_image=False,
                clip_input_patch=False
                )() 


            print(f"Prediction shape: {prediction.shape}")


            # Save the predictions
        
            save_name=os.path.basename(vol)
            save_name = save_name.replace(".mgz","")
            save_name = save_name.replace(".nii.gz","")
            save_name = save_name.replace(".nii","")
            save_name = save_name.replace(".mgh","")
            
            if (mask_list is not None):
                if (mask_list[vol_index] is not None):

                    mask = nib.load(mask_list[vol_index]).get_fdata()
                    masked_count = np.count_nonzero((mask == 0) & (prediction > 0))
                    print('voxel masked out in prediction: ', masked_count)
                    prediction[mask == 0] = 0
            
            if save_native_space:
                print("Saving prediction in native space")
                affine_save = affine
            else:
                print("Saving prediction with identity affine")
                affine_save = np.eye(4)

        

            save_img = nib.Nifti1Image(np.squeeze(prediction), affine=affine_save)
            nib.save(save_img,f"{outputdir}/{save_name}_vessels_prob.mgz")    
            
            if threshold is not None:
                for th in threshold:
                    print(f"Applying threshold: {th}")
                    prediction_binary = (prediction > th).astype(np.float32)
                    save_img = nib.Nifti1Image(np.squeeze(prediction_binary), affine=affine_save)
                    nib.save(save_img,f"{outputdir}/{save_name}_vessels_binary_th_{th}.mgz")
            else:
                print("No threshold applied, saving only raw prediction")
            
        
        t2 = time.time()
        print(f"Process took {round((t2-t1)/60, 2)} min")
