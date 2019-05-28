#!/bin/bash

#====== parameters ======#
dataset=hmdb_ucf # hmdb_ucf | hmdb_ucf_small | ucf_olympic
class_file='data/classInd_'$dataset'.txt'
training=true # true | false
testing=false # true | false
modality=RGB 
frame_type=feature # frame | feature
num_segments=5 # sample frame # of each video for training
test_segments=5
baseline_type=video
frame_aggregation=trn-m # method to integrate the frame-level features (avgpool | trn | trn-m | rnn | temconv)
add_fc=1
fc_dim=512
arch=resnet101
use_target=uSv # none | Sv | uSv
share_params=Y # Y | N

if [ "$use_target" == "none" ] 
then
	exp_DA_name=baseline
else
	exp_DA_name=DA
fi

#====== select dataset ======#
path_data_root=/home/mchen2/dataset/ # depend on users
path_exp_root=action-experiments/ # depend on users

if [ "$dataset" == "hmdb_ucf" ] || [ "$dataset" == "hmdb_ucf_small" ] ||[ "$dataset" == "ucf_olympic" ]
then
	dataset_source=ucf101 # depend on users
	dataset_target=hmdb51 # depend on users
	dataset_val=hmdb51 # depend on users
	num_source=1438 # number of training data (source) 
	num_target=840 # number of training data (target)

	path_data_source=$path_data_root$dataset_source'/'
	path_data_target=$path_data_root$dataset_target'/'
	path_data_val=$path_data_root$dataset_val'/'

	if [[ "$dataset_source" =~ "train" ]]
	then
		dataset_source=$dataset_source
	else
    	dataset_source=$dataset_source'_train'
	fi

	if [[ "$dataset_target" =~ "train" ]]
	then
		dataset_target=$dataset_target
	else
    	dataset_target=$dataset_target'_train'
	fi
	
	if [[ "$dataset_val" =~ "val" ]]
	then
		dataset_val=$dataset_val
	else
    	dataset_val=$dataset_val'_val'
	fi

	train_source_list=$path_data_source'list_'$dataset_source'_'$dataset'-'$frame_type'.txt'
	train_target_list=$path_data_target'list_'$dataset_target'_'$dataset'-'$frame_type'.txt'
	val_list=$path_data_val'list_'$dataset_val'_'$dataset'-'$frame_type'.txt'

	path_exp=$path_exp_root'Testexp'
fi

pretrained=none

#====== parameters for algorithms ======#
# parameters for DA approaches
dis_DA=none # none | DAN | JAN
alpha=0 # depend on users

adv_pos_0=Y # Y | N (discriminator for relation features)
adv_DA=RevGrad # none | RevGrad
beta_0=0.75 # depend on users
beta_1=0.75 # depend on users
beta_2=0.5 # depend on users

use_attn=TransAttn # none | TransAttn | general
n_attn=1
use_attn_frame=none # none | TransAttn | general

use_bn=none # none | AdaBN | AutoDIAL
add_loss_DA=attentive_entropy # none | target_entropy | attentive_entropy
gamma=0.003 # depend on users

ens_DA=none # none | MCD
mu=0

# parameters for architectures
bS=128 # batch size
bS_2=$((bS * num_target / num_source ))
echo '('$bS', '$bS_2')'

lr=3e-2
optimizer=SGD

if [ "$use_target" == "none" ] 
then
	dis_DA=none
	alpha=0
	adv_pos_0=N
	adv_DA=none
	beta_0=0
	beta_1=0
	beta_2=0
	use_attn=none
	use_attn_frame=none
	use_bn=none
	add_loss_DA=none
	gamma=0
	ens_DA=none
	mu=0
	j=0

	exp_path=$path_exp'-'$optimizer'-share_params_'$share_params'/'$dataset'-'$num_segments'seg_'$j'/'
else
	exp_path=$path_exp'-'$optimizer'-share_params_'$share_params'-lr_'$lr'-bS_'$bS'_'$bS_2'/'$dataset'-'$num_segments'seg-disDA_'$dis_DA'-alpha_'$alpha'-advDA_'$adv_DA'-beta_'$beta_0'_'$beta_1'_'$beta_2'-useBN_'$use_bn'-addlossDA_'$add_loss_DA'-gamma_'$gamma'-ensDA_'$ens_DA'-mu_'$mu'-useAttn_'$use_attn'-n_attn_'$n_attn'/'
fi

echo 'exp_path: '$exp_path


#====== select mode ======#
if ($training) 
then
	
	val_segments=$test_segments

	# parameters for optimization
	lr_decay=10
    	lr_adaptive=dann # none | loss | dann
    	lr_steps_1=10
    	lr_steps_2=20
    	epochs=30
	gd=20

	# other parameters (still in progress)	
	pred_normalize=N
	weighted_class_loss_DA=N
	weighted_class_loss=N
	
	#------ main command ------#
	python main.py $dataset $class_file $modality $train_source_list $train_target_list $val_list --exp_path $exp_path \
	--arch $arch --pretrained $pretrained --baseline_type $baseline_type --frame_aggregation $frame_aggregation \
	--num_segments $num_segments --val_segments $val_segments --add_fc $add_fc --fc_dim $fc_dim --dropout_i 0.5 --dropout_v 0.5 \
	--use_target $use_target --share_params $share_params \
	--dis_DA $dis_DA --alpha $alpha --place_dis N Y N \
	--adv_DA $adv_DA --beta $beta_0 $beta_1 $beta_2 --place_adv $adv_pos_0 Y Y \
	--use_bn $use_bn --add_loss_DA $add_loss_DA --gamma $gamma \
	--ens_DA $ens_DA --mu $mu \
	--use_attn $use_attn --n_attn $n_attn --use_attn_frame $use_attn_frame \
	--pred_normalize $pred_normalize --weighted_class_loss_DA $weighted_class_loss_DA --weighted_class_loss $weighted_class_loss \
	--gd $gd --lr $lr --lr_decay $lr_decay --lr_adaptive $lr_adaptive --lr_steps $lr_steps_1 $lr_steps_2 --epochs $epochs --optimizer $optimizer \
	--n_rnn 1 --rnn_cell LSTM --n_directions 1 --n_ts 5 \
	-b $bS $bS_2 $bS -j 4 -ef 1 -pf 50 -sf 50 --copy_list N N --save_model \

fi

if ($testing)
then
	model=model_best # checkpoint | model_best
	echo $model

	# testing on the validation set
	echo 'testing on the validation set'
	python test_models.py $class_file $modality \
	$val_list $exp_path$modality'/'$model'.pth.tar' \
	--arch $arch --test_segments $test_segments \
	--save_scores $exp_path$modality'/scores_'$dataset_target'-'$model'-'$test_segments'seg' --save_confusion $exp_path$modality'/confusion_matrix_'$dataset_target'-'$model'-'$test_segments'seg' \
	--n_rnn 1 --rnn_cell LSTM --n_directions 1 --n_ts 5 \
	--use_attn $use_attn --n_attn $n_attn --use_attn_frame $use_attn_frame --use_bn $use_bn --share_params $share_params \
	-j 4 --bS 512 --top 1 3 5 --add_fc 1 --fc_dim $fc_dim --baseline_type $baseline_type --frame_aggregation $frame_aggregation 

fi

# ----------------------------------------------------------------------------------
exit 0
