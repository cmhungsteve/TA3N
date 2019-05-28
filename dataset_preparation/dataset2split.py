# split the dataset into train/val sets (randomly choose files according to the split ratio)
import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(description='Split the dataset')
parser.add_argument('--data_path', default='data/', help='data path', type=str, required=False)
parser.add_argument('--folder_in', default='folder_in/', help='input folder', type=str, required=False)
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--folder_out_1', default='folder_out_1/', help='1st output folder', type=str, required=False)
parser.add_argument('--folder_out_2', default='folder_out_2/', help='2nd output folder', type=str, required=False)
parser.add_argument('--input_type', type=str, default='video', choices=['video', 'frames'], help='input types for videos')
parser.add_argument('--split_ratio', type=float, required=False, default=0.7, help='ratio of train/val for each class')
parser.add_argument('--split_feat', default='N', help='split the feature vectors as well', type=str, required=False)
args = parser.parse_args()

path_input = args.data_path + args.folder_in # input videos
path_output_1 = args.data_path + args.folder_out_1
path_output_2 = args.data_path + args.folder_out_2

# create folders for split videos (or frames) & features
if not os.path.isdir(path_output_1 + args.modality + '/'):
	print('create', path_output_1 + args.modality + '/')
	os.makedirs(path_output_1 + args.modality + '/')

if not os.path.isdir(path_output_2 + args.modality + '/'):
	print('create', path_output_2 + args.modality + '/')
	os.makedirs(path_output_2 + args.modality + '/')

if args.split_feat == 'Y':
	if not os.path.isdir(path_output_1 + args.modality + '-feature/'):
		print('create', path_output_1 + args.modality + '-feature/')
		os.makedirs(path_output_1 + args.modality + '-feature/')

	if not os.path.isdir(path_output_2 + args.modality + '-feature/'):
		print('create', path_output_2 + args.modality + '-feature/')
		os.makedirs(path_output_2 + args.modality + '-feature/')

################### Main Function ###################
def copy_files(class_name, files, path):
	#--- path ---#
	destination = "{}{}/{}/".format(path, args.modality, class_name) # destination location
	destination_feature = "{}{}-feature/".format(path, args.modality)  # destination location for features
	# remove old images
	if os.path.exists(destination):
		print("deleted old {}".format(destination))
		shutil.rmtree(destination)
	os.makedirs(destination)

	#--- copy files/folders ---#
	for file in files:
		# frames/video
		path_origin = "{}{}/{}/{}".format(path_input, args.modality, class_name, file) # origin location
		if args.input_type == 'video':
			shutil.copyfile(path_origin, destination + file)
		elif args.input_type == 'frames':
			shutil.copytree(path_origin, destination + file)

		# features
		if args.split_feat == 'Y':
			file_name = file.split('.')[0]
			path_origin_feature = "{}{}-feature/{}".format(path_input, args.modality, file_name)  # origin location for features
			shutil.copytree(path_origin_feature, destination_feature + file_name)

################### Main Program ###################
list_class = os.listdir(path_input + args.modality)
list_class.sort()

for class_dir in list_class:
	files = os.listdir("{}{}/{}".format(path_input, args.modality, class_dir))  # all the files in this class
	num_files = len(files)  # file #

	files = set(files)  # convert from list to set

	if args.split_ratio < 0: # split the training/validation sets based on the text file
		if args.folder_in == 'olympic/':
			files_set1 = [line.strip() + '.avi' for line in open(path_input + 'train/' + class_dir + '.txt')]
			files_set2 = [line.strip() + '.avi' for line in open(path_input + 'test/' + class_dir + '.txt')]
			num_files_1 = len(files_set1)
			num_files_2 = len(files_set2)

		else:
			raise ValueError('The dataset is not listed!!')

	else:
		num_files_1 = max(int(num_files*args.split_ratio),1)
		num_files_2 = num_files - num_files_1
		files_set1 = set(random.sample(files, num_files_1))
		files_set2 = files - files_set1

	print(class_dir + ": {}".format(num_files) + " --> {}/{}".format(num_files_1, num_files_2))

	copy_files(class_dir, files_set1, path_output_1)
	copy_files(class_dir, files_set2, path_output_2)
