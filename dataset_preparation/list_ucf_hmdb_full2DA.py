# generate the file list from video dataset
import argparse
import os
from colorama import init
from colorama import Fore, Back, Style
init(autoreset=True)

###### Flags ######
parser = argparse.ArgumentParser(description='select DA classes for UCF/HMDB')
parser.add_argument('dataset', type=str, help='ucf101 | hmdb51')
parser.add_argument('modality', type=str, help='rgb | flow')
parser.add_argument('--sp', default=1, help='split #', type=int, required=False)
parser.add_argument('--data_path', default='data/', help='data path', type=str, required=False)
parser.add_argument('--frame_in', default='RGB-feature', help='video frame/feature folder name', type=str, required=False)
parser.add_argument('--method_read', default='video', type=str, choices=['video','frame'], help='approach to load data')
parser.add_argument('--class_file', type=str, default='class.txt', help='process the classes only in the class_file', required=True)
parser.add_argument('--suffix', default=None, help='additional string for filename', type=str, required=False)

args = parser.parse_args()

###### Function for list generation ######
def gen_list_DA(path_input_list, class_indices_DA, class_names_DA, list_type):
	path_output = args.data_path + args.dataset + '/' + 'list_' + args.dataset + '_' + list_type + args.suffix + '.txt'
	file_write = open(path_output,'w')
	class_indices_DA_unique = list(set(class_indices_DA))
	count_video = [0 for i in range(len(class_indices_DA_unique))]
	for line in open(path_input_list):
		# 1. parse [path, length, class_id]
		path_video, len_video, id_video = line.strip().split(' ')
		path_dataset, frame_in, name_video = path_video.rsplit('/', 2)

		# print(path_video, len_video, id_video)
		# print(path_dataset, frame_in, name_video)
		# exit()

		check_class = False

		if args.dataset == 'hmdb51':
			name_video_short = name_video.rsplit('_',6)[0] # remove the suffix
			name_str = name_video_short.rsplit('_',2)[-2:] # remove the prefix
			class_str = '_'.join(name_str) # join the strings
		
			if class_str.split('_')[1] in class_names_DA:
				check_class = True
				id_video_DA = class_indices_DA[class_names_DA.index(class_str.split('_')[1])]
			elif class_str in class_names_DA:
				check_class = True
				id_video_DA = class_indices_DA[class_names_DA.index(class_str)]
		
		elif args.dataset == 'ucf101':
			class_str = name_video.split('_')[1]
			if class_str in class_names_DA:
				check_class = True
				id_video_DA = class_indices_DA[class_names_DA.index(class_str)]

		if check_class:
			# 2. rearrange and write a new line
			count_video[class_indices_DA_unique.index(id_video_DA)] += 1

			if args.method_read == 'frame':
				len_video = str(len(os.listdir(path_dataset + '/' + args.frame_in  + '/' + name_video)))

			line_new = path_dataset + '/' + args.frame_in  + '/' + name_video + ' ' + len_video + ' ' + str(id_video_DA) + '\n'
			file_write.write(line_new)

	file_write.close()

	# print the video # in each class
	print(path_output)
	for j in range(len(count_video)):
		print(class_indices_DA_unique[j], count_video[j])

###### data path ######
print(Fore.GREEN + 'dataset:', args.dataset)
path_split_folder = args.data_path + args.dataset + '/' + args.dataset + '_splits/'
path_train_list = path_split_folder + args.dataset + '_' + args.modality + '_train_split_' + str(args.sp) + '.txt'
path_val_list = path_split_folder + args.dataset + '_' + args.modality + '_val_split_' + str(args.sp) + '.txt'

###### Load the class list ######
class_indices = [int(line.strip().split(' ', 1)[0]) for line in open(args.class_file)]
class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
# print(class_indices)
# print(class_names)
# exit()

###### Generate lists for DA ######
print(Fore.CYAN + 'generating lists......')
gen_list_DA(path_train_list, class_indices, class_names, 'train')
gen_list_DA(path_val_list, class_indices, class_names, 'val')
