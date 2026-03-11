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
import math
import zarr
from ome_zarr.io import parse_url





if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Method for vessel segmentation.')

    parser.add_argument('-i', '--inpvol', type=str, nargs='+', required=True,
                        help='path to volumes you want to predict on. It can be a list of volumes.') 
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help='output directory to save predictions.')
    parser.add_argument('-mod', '--modality', type=str, required=True,
                        help='Type of modality. Allowed: T2star, HipCT, OCT, TOF, LSFM, fibers.')
    parser.add_argument('-t', '--threshold', type=float, nargs='+', default=[0.3],
                        help='Threshold to apply to the predictions. Default is 0.3. It can be a list of thresholds.')
    parser.add_argument('-m', '--masks', type=str, nargs='+', default=None,
                        help='List of masks to apply to the predictions. If None, no mask is applied. \
                            Masks should be in the same order as the volumes. Default is None.')
    parser.add_argument('-zc', '--zarr_cutout', type=str, nargs='+', default=None,
                        help='A bounding box to identify ROI for the zarr input, "x1,x2,y1,y2,z1,z2" ')
    parser.add_argument('-nw', '--no_weights', action='store_true',
                        help='Whether not to use patch weighting during prediction. Default is using weights.')

   
    #parameters of volumes to predict on
    
    # parser.add_argument('--save-native-space', action='store_true',
    #                     help='Whether to also save the predictions in native space. Default is False.')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='size of UNet input (and size of sliding prediction patch). Default is 128.')
    # parser.add_argument('--step-size', type=int, default=32,
    #                     help='step size (in vx) between adjacent prediction patches. Default is 32.')

    
    
    args = parser.parse_args()
    
    
    volumes = args.inpvol
    outputdir = args.outdir
    modality = args.modality
    threshold = args.threshold
    mask_list = args.masks
    zarr_cutout = args.zarr_cutout
    use_weights = not args.no_weights
    patch_size = args.patch_size
    step_size = patch_size // 4  #fixed step size as quarter of patch size

    #hardcoded parameters
    #patch_size = 128
    #step_size = 32
    final_activation = 'Sigmoid'
    save_native_space = True
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

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


    model_path = './Vessynth/models/'
    
    if modality == 'OCT':
        model_to_load = glob.glob(model_path + 'weights/OCT_model*')[0]
        json_path = os.path.join(model_path, f'segnet_model_OCT.json') #json file containing backbone info
    elif modality == 'T2star':
        #model_to_load = glob.glob('./models/weights/T2star_model4*')[0]
        model_to_load = glob.glob(model_path + 'weights/T2star_model23*')[0]
        json_path = os.path.join(model_path, f'segnet_model_T2star.json')
    elif modality == 'TOF':
        model_to_load = glob.glob(model_path + 'weights/TOF_model54*')[0]
        json_path = os.path.join(model_path, f'segnet_model_TOF.json')
    elif modality == 'HipCT':
        model_to_load = glob.glob(model_path + 'weights/HipCT_model14_epoch100*')[0]
        json_path = os.path.join(model_path, f'segnet_model_HipCT.json')
    elif modality == 'fibers':    
        model_to_load = glob.glob(model_path + 'weights/model3_epoch206*')[0]
        json_path = os.path.join(model_path, f'segnet_model_fibers.json')
    elif modality == 'LSFM':
        model_to_load = glob.glob(model_path + 'weights/LSFM_model14*')[0]
        json_path = os.path.join(model_path, f'segnet_model_LSFM.json')    
    else:
        raise ValueError('Modality not recognized. Allowed: OCT, T2star, HipCT, TOF, LSFM, fibers.')
    

    if not os.path.exists(outputdir):
        print(f"Creating output directory: {outputdir}")
        os.makedirs(outputdir)


    t1 = time.time()
    with torch.no_grad():

        
        # Read backbone_dict from the JSON file
        with open(json_path, 'r') as f:
            backbone_dict = json.load(f)
        print("\nLoaded model backbone info from", json_path)
        
        model = SegNet(ndim=3, in_channels=1, out_channels=1,
                        init_kernel_size=3, final_activation=final_activation, backbone='UNet', 
                        kwargs_backbone=backbone_dict)

        

        print('Loading model: ', model_to_load)

        saved_model = torch.load(model_to_load, map_location=torch.device(DEVICE))
        #saved_model = torch.load(model_to_load) # for gpu
        
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

            save_name=os.path.basename(vol)
            save_name = save_name.replace(".mgz","")
            save_name = save_name.replace(".nii.gz","")
            save_name = save_name.replace(".nii","")
            save_name = save_name.replace(".mgh","")
            
            split_up = False
            X = 1
            Y = 1
            Z = 1
            cut_size = 1024
            if len(outputdir) > len(".zarr") and outputdir[-len(".zarr"):] == ".zarr" and zarr_cutout is not None:
                split_up = True
                zarr_split = zarr_cutout[0].split(",")
                X =  int(math.ceil((int(zarr_split[1]) - int(zarr_split[0]))/cut_size))
                Y =  int(math.ceil((int(zarr_split[3]) - int(zarr_split[2]))/cut_size))
                Z =  int(math.ceil((int(zarr_split[5]) - int(zarr_split[4]))/cut_size))

                store = parse_url(outputdir, mode="a").store

                root = zarr.group(store=store, overwrite=True)

                zarr_dataset = root.create_dataset(
                    "prob",
                    shape=(
                        int(zarr_split[1]) - int(zarr_split[0]),
                        int(zarr_split[3]) - int(zarr_split[2]),
                        int(zarr_split[5]) - int(zarr_split[4])
                    ),
                    chunks=(128, 128, 128),
                    dtype="float32"
                )
                if threshold is not None:
                    zarr_dataset_thresh = []
                    for th in threshold:
                        zarr_dataset_thresh.append(root.create_dataset(
                            f"threshold_{th}",
                            shape=(
                                int(zarr_split[1]) - int(zarr_split[0]),
                                int(zarr_split[3]) - int(zarr_split[2]),
                                int(zarr_split[5]) - int(zarr_split[4])
                            ),
                            chunks=(128, 128, 128),
                            dtype="float32"
                        ))


            slice_ind = 0
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        zarr_tmp = zarr_cutout
                        if split_up:
                            slice_ind += 1
                            print("running prediction slice ", slice_ind, "/", X*Y*Z)
                            zarr_cutout_split = zarr_cutout[0].split(",")
                            zarr_cutout_split[1] = str(min(cut_size*(x+1) + int(zarr_cutout_split[0]), int(zarr_cutout_split[1])))
                            zarr_cutout_split[0] = str(min(cut_size*x + int(zarr_cutout_split[0]), int(zarr_cutout_split[1])))
                            zarr_cutout_split[3] = str(min(cut_size*(y+1) + int(zarr_cutout_split[2]), int(zarr_cutout_split[3])))
                            zarr_cutout_split[2] = str(min(cut_size*y + int(zarr_cutout_split[2]), int(zarr_cutout_split[3])))
                            zarr_cutout_split[5] = str(min(cut_size*(z+1) + int(zarr_cutout_split[4]), int(zarr_cutout_split[5])))
                            zarr_cutout_split[4] = str(min(cut_size*z + int(zarr_cutout_split[4]), int(zarr_cutout_split[5])))
                            zarr_tmp = [",".join(zarr_cutout_split)]
                        prediction, affine = test_convolve(
                            vol,
                            model,
                            patch_size,
                            step_size,
                            DEVICE=DEVICE,
                            normalize_patches=True, 
                            normalize_image=False,
                            clip_input_patch=False,
                            cutout=zarr_tmp,
                            use_weights=use_weights
                            )() 
                        
                        if split_up:
                            print("saving prediction to zarr file")
                            sliceX = [int(zarr_cutout_split[0]) - int(zarr_split[0]), int(zarr_cutout_split[1]) - int(zarr_split[0])]
                            sliceY = [int(zarr_cutout_split[2]) - int(zarr_split[2]), int(zarr_cutout_split[3]) - int(zarr_split[2])]
                            sliceZ = [int(zarr_cutout_split[4]) - int(zarr_split[4]), int(zarr_cutout_split[5]) - int(zarr_split[4])]
                            region = (slice(sliceX[0], sliceX[1]), slice(sliceY[0], sliceY[1]), slice(sliceZ[0], sliceZ[1]))
                            zarr_dataset[sliceX[0]:sliceX[1], sliceY[0]:sliceY[1], sliceZ[0]:sliceZ[1]] = prediction
                            if threshold is not None:
                                for th_idx in range(len(threshold)):
                                    prediction_binary = (prediction > threshold[th_idx]).astype(np.float32)
                                    zarr_dataset_thresh[th_idx][sliceX[0]:sliceX[1], sliceY[0]:sliceY[1], sliceZ[0]:sliceZ[1]] = prediction_binary
                                    del prediction_binary
                        
                        else:
                            if save_native_space:
                                print("Saving prediction in native space")
                                affine_save = affine
                            else:
                                print("Saving prediction with identity affine")
                                affine_save = np.eye(4)


                            if (mask_list is not None):
                                if (mask_list[vol_index] is not None):

                                    mask = nib.load(mask_list[vol_index]).get_fdata()
                                    masked_count = np.count_nonzero((mask == 0) & (prediction > 0))
                                    print('voxel masked out in prediction: ', masked_count)
                                    save_img = nib.Nifti1Image(np.squeeze(prediction), affine=affine_save)
                                    nib.save(save_img,f"{outputdir}/{save_name}_vessels_prob_unmasked.nii.gz")
                                    prediction[mask == 0] = 0

                            save_img = nib.Nifti1Image(np.squeeze(prediction), affine=affine_save)
                            nib.save(save_img,f"{outputdir}/{save_name}_vessels_prob.nii.gz")    
                            
                            if threshold is not None:
                                for th in threshold:
                                    print(f"Applying threshold: {th}")
                                    prediction_binary = (prediction > th).astype(np.float32)
                                    save_img = nib.Nifti1Image(np.squeeze(prediction_binary), affine=affine_save)
                                    nib.save(save_img,f"{outputdir}/{save_name}_vessels_binary_th_{th}.nii.gz")
                                    del prediction_binary

                            else:
                                print("No threshold applied, saving only raw prediction")
                    
                        del prediction
        t2 = time.time()
        print(f"Process took {round((t2-t1)/60, 2)} min")
