#!/bin/bash
# ----------------------------------------------------------------------------------
# variable
dataset=ucf101 # hmdb51 | ucf101
modality=rgb # rgb | flow
sp=1 # 1 | 2 | 3
data_path=/dataset/ # depend on users
frame_type=feature # frame | feature
method_read=frame # frame | video (much slower)
DA_setting=hmdb_ucf # hmdb_ucf | hmdb_ucf_small
frame_in='RGB-'$frame_type
class_file='../data/'$dataset'_splits/class_list_'$DA_setting'.txt'
suffix='_'$DA_setting'-'$frame_type

python list_ucf_hmdb_full2DA.py $dataset $modality --sp $sp --data_path $data_path \
--frame_in $frame_in --class_file $class_file --suffix $suffix --method_read $method_read

#----------------------------------------------------------------------------------
exit 0
