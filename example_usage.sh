basePath=/autofs/space/mulan_002/users/chiara/vessels/
dataPath=$basePath'all_patches_for_paper/Hip-CT/original/' 
outputDir=$basePath'HipCT/output_diceloss_mytrain/predictions_chuncks/model1_epoch68/'
subjects=(
    LADAF-2021-17-chunk1_smaller.mgz
    LADAF-2021-17-chunk2_smaller.mgz
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

python vessynth_test.py -i "${dataPath}${subjects[0]}" -o $outputDir -mod HipCT 
python vessynth_test.py -i "${dataPath}${subjects[1]}" -o $outputDir -mod HipCT



















