#!/bin/bash
# ----------------------------------------------------------------------------------
# variable

data_path=/home/mchen2/dataset/PS/unlabeled/ # depend on users
out_folder=video_frames # depend on users

python -W ignore video_processing.py --video_in Fortnite_20180411162003.mp4 --data_path $data_path -w $out_folder

#----------------------------------------------------------------------------------
exit 0
