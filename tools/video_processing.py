import argparse
from colorama import init
from colorama import Fore, Back, Style
import os
import imageio
import numpy as np
from tqdm import tqdm

import time

import cv2 # add texts and draw figures on frames

imageio.plugins.ffmpeg.download()
init(autoreset=True)

###### Flags ######
parser = argparse.ArgumentParser(description='Video Processing Code')
# data
parser.add_argument('--data_path', type=str, required=False, default='', help='source path')
parser.add_argument('--video_in', type=str, required=False, default='test.mp4', help='name of the input video')
# others
parser.add_argument('--verbose', default=False, action="store_true")
parser.add_argument('-w', '--write_video', type=str, required=False, default='video_output', help='name of the output folder')

args = parser.parse_args()

###### data path ######
path_video_in = args.data_path + args.video_in

#--- create the output folder
path_output = args.data_path + args.write_video + '/'
if not os.path.isdir(path_output) and args.write_video:
	os.makedirs(path_output)

###### Prepare the video i/o ######
# meta_data: nframes, fps, size, duration
print('video:', path_video_in)
reader = imageio.get_reader(path_video_in)
num_frames_total = reader.get_meta_data()['nframes']
fps = reader.get_meta_data()['fps']
video_name = args.video_in.split('.')[0]
if args.write_video:
	writer = imageio.get_writer(path_output + video_name + '_proc.mp4', fps=fps)

###### Video Processing ######
start = time.time()
print(Fore.CYAN + 'Process the video......')
size = cv2.getTextSize('Frame: ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
try:
	for t, im in tqdm(enumerate(reader)):
		if np.sum(im.shape) != 0:
			im_new = im
			line = 'Frame: ' + str(t)
			if args.write_video:
				cv2.putText(im_new, line, (10, int(size[1] * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
				writer.append_data(im_new)

except RuntimeError:
	print(Back.RED + 'Could not read frame', t + 1, 'from', args.video_in)

if args.write_video:
	writer.close()

end = time.time()
print('Total elapsed time: ' + str(end - start))