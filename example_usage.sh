basePath=/autofs/space/mulan_002/users/chiara/vessels/
dataPath=$basePath'all_patches_for_paper/exvivoMRI/original/cortex-subcort/' 
outputDir=$basePath/tests_code/
subjects=(
    I53_bias_corrected.mgz
    I61_flash20_RAS.mgz
)

# mask='given'   #'given','compute' or 'none' (default: 'given') ####


#TEST PREDICTIONS
# echo " "
# echo "Running predictions"
# saveDirName='hemis_predictions'
# inputVolumes=()
# for subject in "${subjects[@]}"; do
#     inputVolumes+=("${dataPath}${subject}")
#     if [[ "$mask" == "given" ]]; then
#         subject_ID="${subject%%_*}"
#         maskPath=$dataPath  ####
#         maskName="${subject_ID}_binary_mask_open_sulci_LIA.mgz" ####
#         masks+=("${maskPath}${maskName}")
#     elif [[ "$mask" == "compute" ]]; then
#         subject_ID="${subject%%_*}"
#         mri_synthstrip -i "${dataPath}${subject}" -o "${dataPath}${subject_ID}_stripped.mgz" -m "${dataPath}${subject_ID}_mask.mgz"
#         masks+=("${maskPath}${subject_ID}_mask.mgz")
#     else
#         masks+=('None')
#     fi
# done

#echo "${inputVolumes[@]}"

#predict on all patches

python vessynth_test.py -i "${dataPath}${subjects[0]}" "${dataPath}${subjects[1]}" -o $outputDir -m T2star 



















