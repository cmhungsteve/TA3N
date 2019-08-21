#!/bin/bash
# ----------------------------------------------------------------------------------
# variable
data_path=/dataset/ # depend on users
folder_in=olympic/ # depend on users
modality=RGB
folder_out_1=olympic_train/ # depend on users
folder_out_2=olympic_val/ # depend on users
input_type=video # frames | video
split_ratio=0.8 # <0: load the split text files
split_feat=Y # Y (need to generate all the features first) | N 

python dataset2split.py --data_path $data_path --folder_in $folder_in --modality $modality \
--folder_out_1 $folder_out_1 --folder_out_2 $folder_out_2 --input_type $input_type \
--split_ratio $split_ratio --split_feat $split_feat

#----------------------------------------------------------------------------------
exit 0
