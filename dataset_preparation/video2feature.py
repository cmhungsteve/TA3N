import argparse
from colorama import init
from colorama import Fore, Back, Style
import os
import imageio
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

import time

# for extracting feature vectors
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

imageio.plugins.ffmpeg.download()
init(autoreset=True)

###### Flags ######
parser = argparse.ArgumentParser(description='Dataset Preparation')
parser.add_argument('--data_path', type=str, required=False, default='', help='source path')
parser.add_argument('--video_in', type=str, required=False, default='RGB', help='name of input video dataset')
parser.add_argument('--feature_in', type=str, required=False, default='RGB-feature', help='name of output frame dataset')
parser.add_argument('--input_type', type=str, default='video', choices=['video', 'frames'], help='input types for videos')
parser.add_argument('--structure', type=str, default='tsn', choices=['tsn', 'imagenet'], help='data structure of output frames')
parser.add_argument('--base_model', type=str, required=False, default='resnet101', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'c3d'])
parser.add_argument('--pretrain_weight', type=str, required=False, default='', help='model weight file path')
parser.add_argument('--num_thread', type=int, required=False, default=-1, help='number of threads for multiprocessing')
parser.add_argument('--batch_size', type=int, required=False, default=1, help='batch size')
parser.add_argument('--start_class', type=int, required=False, default=1, help='the starting class id (start from 1)')
parser.add_argument('--end_class', type=int, required=False, default=-1, help='the end class id')
parser.add_argument('--class_file', type=str, default='class.txt', help='process the classes only in the class_file')

args = parser.parse_args()

# Create thread pool
max_thread = 8 # there are some issues if too many threads
num_thread = args.num_thread if args.num_thread>0 and args.num_thread<=max_thread else max_thread
print(Fore.CYAN + 'thread #:', num_thread)
pool = ThreadPool(num_thread)

###### data path ######
path_input = args.data_path + args.video_in + '/'
feature_in_type = '.t7'

#--- create dataset folders
# root folder
path_output = args.data_path + args.feature_in + '_' + args.base_model + '/'
if args.structure != 'tsn':
	path_output = args.data_path + args.feature_in + '-' + args.structure + '/'
if not os.path.isdir(path_output):
	os.makedirs(path_output)

###### set up the model ######
# Load the pretrained model
print(Fore.GREEN + 'Pre-trained model:', args.base_model)

if args.base_model == 'c3d':
	from C3D_model import C3D
	c3d_clip_size = 16
	model = C3D()
	model.load_state_dict(torch.load(args.pretrain_weight))
    
	list_model = list(model.children())
	list_conv = list_model[:-6]
	list_fc = list_model[-6:-4]
	extractor_conv = nn.Sequential(*list_conv)
	extractor_fc = nn.Sequential(*list_fc)

	# multi-gpu
	extractor_conv = torch.nn.DataParallel(extractor_conv.cuda())
	extractor_conv.eval()
	extractor_fc = torch.nn.DataParallel(extractor_fc.cuda())
	extractor_fc.eval()

else:
	model = getattr(models, args.base_model)(pretrained=True)
	# remove the last layer
	feature_map = list(model.children())
	feature_map.pop()
	extractor = nn.Sequential(*feature_map)
	# multi-gpu
	extractor = torch.nn.DataParallel(extractor.cuda())
	extractor.eval()

cudnn.benchmark = True

#--- data pre-processing
if args.base_model == 'c3d':
	data_transform = transforms.Compose([
		transforms.Resize(112),
		transforms.CenterCrop(112),
		transforms.ToTensor(),
		])
else:
	data_transform = transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

# read the class files
if args.class_file == 'none':
	class_names_proc = ['unlabeled']
else:
	class_names_proc = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]

################### Main Function ###################
def im2tensor(im):
	im = Image.fromarray(im) # convert numpy array to PIL image
	t_im = data_transform(im) # Create a PyTorch Variable with the transformed image
	return t_im

def extract_frame_feature_batch(list_tensor):
	with torch.no_grad():
		batch_tensor = torch.stack(list_tensor)

		if args.base_model == 'c3d':
			batch_tensor = convert_c3d_tensor_batch(batch_tensor) # e.g. 113x3x16x112x112
			batch_tensor = Variable(batch_tensor).cuda() # Create a PyTorch Variable
			features_conv = extractor_conv(batch_tensor) # e.g. 113x512x1x4x4
			features_conv = features_conv.view(features_conv.size(0),-1) # e.g. 113x8192
			features = extractor_fc(features_conv)
		else:
			batch_tensor = Variable(batch_tensor).cuda() # Create a PyTorch Variable
			features = extractor(batch_tensor)
		features = features.view(features.size(0), -1).cpu()
		return features

def convert_c3d_tensor_batch(batch_tensor): # e.g. 30x3x112x112 --> 15x3x16x112x112
	batch_tensor_c3d = torch.Tensor()
	for b in range(batch_tensor.size(0)-c3d_clip_size+1):
		tensor_c3d = batch_tensor[b:b+c3d_clip_size,:,:,:]
		tensor_c3d = torch.transpose(tensor_c3d,0,1).unsqueeze(0)
		batch_tensor_c3d = torch.cat((batch_tensor_c3d, tensor_c3d))

	batch_tensor_c3d = batch_tensor_c3d*255
	return batch_tensor_c3d

def extract_features(video_file):
	print(video_file)
	video_name = os.path.splitext(video_file)[0]
	if args.structure == 'tsn':  # create the video folder if the data structure is TSN
		if not os.path.isdir(path_output + video_name + '/'):
			os.makedirs(path_output + video_name + '/')

	num_exist_files = len(os.listdir(path_output + video_name + '/'))

	frames_tensor = []
	# print(class_name)
	if args.input_type == 'video':
		reader = imageio.get_reader(path_input + class_name + '/' + video_file)

		#--- collect list of frame tensors
		try:
			for t, im in enumerate(reader):
				if np.sum(im.shape) != 0:
					id_frame = t+1
					frames_tensor.append(im2tensor(im))  # include data pre-processing
		except RuntimeError:
			print(Back.RED + 'Could not read frame', id_frame+1, 'from', video_file)
	elif args.input_type == 'frames':
		list_frames = os.listdir(path_input + class_name + '/' + video_file)
		list_frames.sort()

		# --- collect list of frame tensors
		try:
			for t in range(len(list_frames)):
				im = imageio.imread(path_input + class_name + '/' + video_file + '/' + list_frames[t])
				if np.sum(im.shape) != 0:
					id_frame = t+1
					frames_tensor.append(im2tensor(im))  # include data pre-processing
		except RuntimeError:
			print(Back.RED + 'Could not read frame', id_frame+1, 'from', video_file)


	#--- divide the list into two parts: major (can de divided by batch size) & the rest (will add dummy tensors)
	num_frames = len(frames_tensor)
	if num_frames == num_exist_files: # skip if the features are already saved
		return

	num_major = num_frames//args.batch_size*args.batch_size
	num_rest = num_frames - num_major

	# add dummy tensor to make total size == batch_size*N
	num_dummy = args.batch_size - num_rest
	for i in range(num_dummy):
		frames_tensor.append(torch.zeros_like(frames_tensor[0]))

	#--- extract video features
	features = torch.Tensor()

	for t in range(0, num_frames+num_dummy, args.batch_size):
		frames_batch = frames_tensor[t:t+args.batch_size]
		features_batch = extract_frame_feature_batch(frames_batch)
		features = torch.cat((features,features_batch))

	features = features[:num_frames] # remove the dummy part

	#--- save the frame-level feature vectors to files
	for t in range(features.size(0)):
		id_frame = t+1
		id_frame_name = str(id_frame).zfill(5)
		if args.structure == 'tsn':
			filename = path_output + video_name + '/' + 'img_' + id_frame_name + feature_in_type
		elif args.structure == 'imagenet':
			filename = path_output + class_name + '/' + video_name + '_' + id_frame_name + feature_in_type
		else:
			raise NameError(Back.RED + 'not valid data structure')

		if not os.path.exists(filename):
			torch.save(features[t].clone(), filename) # if no clone(), the size of features[t] will be the same as features

################### Main Program ###################
# parse the classes
list_class = os.listdir(path_input)
list_class.sort()

# for i in range(len(list_class)):
# 	print(i, list_class[i])
# exit()

id_class_start = args.start_class-1
id_class_end = len(list_class) if args.end_class <= 0 else args.end_class
start = time.time()

for i in range(id_class_start, id_class_end):
	start_class = time.time()
	class_name = list_class[i]
	if class_name in class_names_proc:
		print(Fore.YELLOW + 'class ' + str(i+1) + ': ' + class_name)

		if args.structure == 'imagenet': # create the class folder if the data structure is ImageNet
			if not os.path.isdir(path_output + class_name + '/'):
				os.makedirs(path_output + class_name + '/')

		list_video = os.listdir(path_input + class_name + '/')
		list_video.sort()

		pool.map(extract_features, list_video, chunksize=1)

		end_class = time.time()
		print('Elapsed time for ' + class_name + ': ' + str(end_class-start_class))
	else:
		print(Fore.RED + class_name + ' is not selected !!')

end = time.time()
print('Total elapsed time: ' + str(end-start))
print(Fore.GREEN + 'All the features are generated for ' + args.video_in)
