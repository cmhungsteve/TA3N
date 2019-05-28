# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from models import VideoModel
from loss import *
from opts import parser
from utils.utils import randSelectBatch
import math

from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()

def main():
	global args, best_prec1, writer
	args = parser.parse_args()

	print(Fore.GREEN + 'Baseline:', args.baseline_type)
	print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

	print(Fore.GREEN + 'target data usage:', args.use_target)
	if args.use_target == 'none':
		print(Fore.GREEN + 'no Domain Adaptation')
	else:
		if args.dis_DA != 'none':
			print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
			if len(args.place_dis) != args.add_fc + 2:
				raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

		if args.adv_DA != 'none':
			print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

		if args.use_bn != 'none':
			print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

	# determine the categories
	class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
	num_class = len(class_names)

	#=== check the folder existence ===#
	path_exp = args.exp_path + args.modality + '/'
	if not os.path.isdir(path_exp):
		os.makedirs(path_exp)

	if args.tensorboard:
		writer = SummaryWriter(path_exp + '/tensorboard')  # for tensorboardX

	#=== initialize the model ===#
	print(Fore.CYAN + 'preparing the model......')
	model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments, 
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params)

	model = torch.nn.DataParallel(model, args.gpus).cuda()

	if args.optimizer == 'SGD':
		print(Fore.YELLOW + 'using SGD')
		optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optimizer == 'Adam':
		print(Fore.YELLOW + 'using Adam')
		optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
	else:
		print(Back.RED + 'optimizer not support or specified!!!')
		exit()

	#=== check point ===#
	start_epoch = 1
	print(Fore.CYAN + 'checking the checkpoint......')
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch'])))
			if args.resume_hp:
				print("=> loaded checkpoint hyper-parameters")
				optimizer.load_state_dict(checkpoint['optimizer'])
		else:
			print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	#--- open log files ---#
	if not args.evaluate:
		if args.resume:
			train_file = open(path_exp + 'train.log', 'a')
			train_short_file = open(path_exp + 'train_short.log', 'a')
			val_file = open(path_exp + 'val.log', 'a')
			val_short_file = open(path_exp + 'val_short.log', 'a')
			train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
			train_short_file.write('========== start: ' + str(start_epoch) + '\n')
			val_file.write('========== start: ' + str(start_epoch) + '\n')
			val_short_file.write('========== start: ' + str(start_epoch) + '\n')
		else:
			train_short_file = open(path_exp + 'train_short.log', 'w')
			val_short_file = open(path_exp + 'val_short.log', 'w')
			train_file = open(path_exp + 'train.log', 'w')
			val_file = open(path_exp + 'val.log', 'w')

		val_best_file = open(args.save_best_log, 'a')

	else:
		test_short_file = open(path_exp + 'test_short.log', 'w')
		test_file = open(path_exp + 'test.log', 'w')

	#=== Data loading ===#
	print(Fore.CYAN + 'loading data......')

	if args.use_opencv:
		print("use opencv functions")

	if args.modality == 'RGB':
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
		data_length = 5

	# calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
	num_source = sum(1 for i in open(args.train_source_list))
	num_target = sum(1 for i in open(args.train_target_list))
	num_val = sum(1 for i in open(args.val_list))

	num_iter_source = num_source / args.batch_size[0]
	num_iter_target = num_target / args.batch_size[1]
	num_max_iter = max(num_iter_source, num_iter_target)
	num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
	num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

	# calculate the weight for each class
	class_id_list = [int(line.strip().split(' ')[2]) for line in open(args.train_source_list)]
	class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
	class_freq = (class_data_counts / class_data_counts.sum()).tolist()

	weight_source_class = torch.ones(num_class).cuda()
	weight_domain_loss = torch.Tensor([1, 1]).cuda()

	if args.weighted_class_loss == 'Y':
		weight_source_class = 1 / torch.Tensor(class_freq).cuda()

	if args.weighted_class_loss_DA == 'Y':
		weight_domain_loss = torch.Tensor([1/num_source_train, 1/num_target_train]).cuda()

	# data loading (always need to load the testing data)
	val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
	val_set = TSNDataSet("", args.val_list, num_dataload=num_val, num_segments=val_segments,
						 new_length=data_length, modality=args.modality,
						 image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
																		  "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
						 random_shift=False,
						 test_mode=True,
						 )
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size[2], shuffle=False,
											 num_workers=args.workers, pin_memory=True)

	if not args.evaluate:
		source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
		source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

		target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
		target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)

	# --- Optimizer ---#
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class).cuda()
		criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()
	else:
		raise ValueError("Unknown loss type")

	if args.evaluate:
		print(Fore.CYAN + 'evaluation only......')
		prec1 = validate(val_loader, model, criterion, num_class, 0, test_file)
		test_short_file.write('%.3f\n' % prec1)
		return

	#=== Training ===#
	start_train = time.time()
	print(Fore.CYAN + 'start training......')
	beta = args.beta
	gamma = args.gamma
	mu = args.mu
	loss_c_current = 999 # random large number
	loss_c_previous = 999 # random large number

	attn_source_all = torch.Tensor()
	attn_target_all = torch.Tensor()

	for epoch in range(start_epoch, args.epochs+1):

		## schedule for parameters
		alpha = 2 / (1 + math.exp(-1 * (epoch) / args.epochs)) - 1 if args.alpha < 0 else args.alpha

		## schedule for learning rate
		if args.lr_adaptive == 'loss':
			adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
		elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
			adjust_learning_rate(optimizer, args.lr_decay)

		# train for one epoch
		loss_c, attn_epoch_source, attn_epoch_target = train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, train_file, train_short_file, alpha, beta, gamma, mu)
		
		if args.save_attention >= 0:
			attn_source_all = torch.cat((attn_source_all, attn_epoch_source.unsqueeze(0)))  # save the attention values
			attn_target_all = torch.cat((attn_target_all, attn_epoch_target.unsqueeze(0)))  # save the attention values

		# update the recorded loss_c
		loss_c_previous = loss_c_current
		loss_c_current = loss_c

		# evaluate on validation set
		if epoch % args.eval_freq == 0 or epoch == args.epochs:
			prec1 = validate(val_loader, model, criterion, num_class, epoch, val_file)

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			line_update = ' ==> updating the best accuracy' if is_best else ''
			line_best = "Best score {} vs current score {}".format(best_prec1, prec1) + line_update
			print(Fore.YELLOW + line_best)
			val_short_file.write('%.3f\n' % prec1)

			best_prec1 = max(prec1, best_prec1)

			if args.tensorboard:
				writer.add_text('Best_Accuracy', str(best_prec1), epoch)

			if args.save_model:
				save_checkpoint({
					'epoch': epoch,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'best_prec1': best_prec1,
					'prec1': prec1,
				}, is_best, path_exp)
	
	end_train = time.time()
	print(Fore.CYAN + 'total training time:', end_train - start_train)
	val_best_file.write('%.3f\n' % best_prec1)

	# --- write the total time to log files ---#
	line_time = 'total time: {:.3f} '.format(end_train - start_train)
	if not args.evaluate:
		train_file.write(line_time)
		train_short_file.write(line_time)
		val_file.write(line_time)
		val_short_file.write(line_time)
	else:
		test_file.write(line_time)
		test_short_file.write(line_time)

	#--- close log files ---#
	if not args.evaluate:
		train_file.close()
		train_short_file.close()
		val_file.close()
		val_short_file.close()
	else:
		test_file.close()
		test_short_file.close()

	if args.tensorboard:
		writer.close()

	if args.save_attention >= 0:
		np.savetxt('attn_source_' + str(args.save_attention) + '.log', attn_source_all.cpu().detach().numpy(), fmt="%s")
		np.savetxt('attn_target_' + str(args.save_attention) + '.log', attn_target_all.cpu().detach().numpy(), fmt="%s")


def train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_a = AverageMeter()  # adversarial loss
	losses_d = AverageMeter()  # discrepancy loss
	losses_e = AverageMeter()  # entropy loss
	losses_s = AverageMeter()  # ensemble loss
	losses_c = AverageMeter()  # classification loss
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	if args.no_partialbn:
		model.module.partialBN(False)
	else:
		model.module.partialBN(True)

	# switch to train mode
	model.train()

	end = time.time()
	data_loader = enumerate(zip(source_loader, target_loader))

	# step info
	start_steps = epoch * len(source_loader)
	total_steps = args.epochs * len(source_loader)

	# initialize the embedding
	if args.tensorboard:
		feat_source_display = None
		label_source_display = None
		label_source_domain_display = None

		feat_target_display = None
		label_target_display = None
		label_target_domain_display = None

	attn_epoch_source = torch.Tensor()
	attn_epoch_target = torch.Tensor()
	for i, ((source_data, source_label),(target_data, target_label)) in data_loader:
		# setup hyperparameters
		p = float(i + start_steps) / total_steps
		beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
		beta = [beta_dann if beta[i] < 0 else beta[i] for i in range(len(beta))] # replace the default beta if value < 0

		source_size_ori = source_data.size()  # original shape
		target_size_ori = target_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		batch_target_ori = target_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < args.batch_size[0]:
			source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
			source_data = torch.cat((source_data, source_data_dummy))
		if batch_target_ori < args.batch_size[1]:
			target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
			target_data = torch.cat((target_data, target_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if source_data.size(0) % gpu_count != 0:
			source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
			source_data = torch.cat((source_data, source_data_dummy))
		if target_data.size(0) % gpu_count != 0:
			target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
			target_data = torch.cat((target_data, target_data_dummy))

		# measure data loading time
		data_time.update(time.time() - end)

		source_label = source_label.cuda(non_blocking=True) # pytorch 0.4.X
		target_label = target_label.cuda(non_blocking=True) # pytorch 0.4.X

		if args.baseline_type == 'frame':
			source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			target_label_frame = target_label.unsqueeze(1).repeat(1, args.num_segments).view(-1)

		label_source = source_label_frame if args.baseline_type == 'frame' else source_label  # determine the label for calculating the loss function
		label_target = target_label_frame if args.baseline_type == 'frame' else target_label

		#====== pre-train source data ======#
		if args.pretrain_source:
			#------ forward pass data again ------#
			_, out_source, out_source_2, _, _, _, _, _, _, _ = model(source_data, target_data, beta, mu, is_train=True, reverse=False)

			# ignore dummy tensors
			out_source = out_source[:batch_source_ori]
			out_source_2 = out_source_2[:batch_source_ori]

			#------ calculate the loss function ------#
			# 1. calculate the classification loss
			out = out_source
			label = label_source

			loss = criterion(out, label)
			if args.ens_DA == 'MCD' and args.use_target != 'none':
				loss += criterion(out_source_2, label)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			if args.clip_gradient is not None:
				total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
				if total_norm > args.clip_gradient and args.verbose:
					print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

			optimizer.step()


		#====== forward pass data ======#
		attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=False)

		# ignore dummy tensors
		attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori)
		attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)

		if args.pred_normalize == 'Y': # use the uncertainly method (in contruction...)
			out_source = out_source / out_source.var().log()
			out_target = out_target / out_target.var().log()

		# store the embedding
		if args.tensorboard:
			feat_source_display = feat_source[1] if i==0 else torch.cat((feat_source_display, feat_source[1]), 0)
			label_source_display = label_source if i==0 else torch.cat((label_source_display, label_source), 0)
			label_source_domain_display = torch.zeros(label_source.size(0)) if i==0 else torch.cat((label_source_domain_display, torch.zeros(label_source.size(0))), 0)
			feat_target_display = feat_target[1] if i==0 else torch.cat((feat_target_display, feat_target[1]), 0)
			label_target_display = label_target if i==0 else torch.cat((label_target_display, label_target), 0)
			label_target_domain_display = torch.ones(label_target.size(0)) if i==0 else torch.cat((label_target_domain_display, torch.ones(label_target.size(0))), 0)

		#====== calculate the loss function ======#
		# 1. calculate the classification loss
		out = out_source
		label = label_source

		if args.use_target == 'Sv':
			out = torch.cat((out, out_target))
			label = torch.cat((label, label_target))

		loss_classification = criterion(out, label)
		if args.ens_DA == 'MCD' and args.use_target != 'none':
			loss_classification += criterion(out_source_2, label)

		losses_c.update(loss_classification.item(), out_source.size(0)) # pytorch 0.4.X
		loss = loss_classification

		# 2. calculate the loss for DA
		# (I) discrepancy-based approach: discrepancy loss
		if args.dis_DA != 'none' and args.use_target != 'none':
			loss_discrepancy = 0

			kernel_muls = [2.0]*2
			kernel_nums = [2, 5]
			fix_sigma_list = [None]*2

			if args.dis_DA == 'JAN':
				# ignore the features from shared layers
				feat_source_sel = feat_source[:-args.add_fc]
				feat_target_sel = feat_target[:-args.add_fc]

				size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
				feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
				feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]

				loss_discrepancy += JAN(feat_source_sel, feat_target_sel, kernel_muls=kernel_muls, kernel_nums=kernel_nums, fix_sigma_list=fix_sigma_list, ver=2)

			else:
				# extend the parameter list for shared layers
				kernel_muls.extend([kernel_muls[-1]]*args.add_fc)
				kernel_nums.extend([kernel_nums[-1]]*args.add_fc)
				fix_sigma_list.extend([fix_sigma_list[-1]]*args.add_fc)

				for l in range(0, args.add_fc + 2):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
					if args.place_dis[l] == 'Y':
						# select the data for calculating the loss (make sure source # == target #)
						size_loss = min(feat_source[l].size(0), feat_target[l].size(0)) # choose the smaller number
						# select
						feat_source_sel = feat_source[l][:size_loss]
						feat_target_sel = feat_target[l][:size_loss]

						# break into multiple batches to avoid "out of memory" issue
						size_batch = min(256,feat_source_sel.size(0))
						feat_source_sel = feat_source_sel.view((-1,size_batch) + feat_source_sel.size()[1:])
						feat_target_sel = feat_target_sel.view((-1,size_batch) + feat_target_sel.size()[1:])

						if args.dis_DA == 'CORAL':
							losses_coral = [CORAL(feat_source_sel[t], feat_target_sel[t]) for t in range(feat_source_sel.size(0))]
							loss_coral = sum(losses_coral)/len(losses_coral)
							loss_discrepancy += loss_coral
						elif args.dis_DA == 'DAN':
							losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l], kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t in range(feat_source_sel.size(0))]
							loss_mmd = sum(losses_mmd) / len(losses_mmd)

							loss_discrepancy += loss_mmd
						else:
							raise NameError('not in dis_DA!!!')

			losses_d.update(loss_discrepancy.item(), feat_source[0].size(0))
			loss += alpha * loss_discrepancy

		# (II) adversarial discriminative model: adversarial loss
		if args.adv_DA != 'none' and args.use_target != 'none':
			loss_adversarial = 0
			pred_domain_all = []
			pred_domain_target_all = []

			for l in range(len(args.place_adv)):
				if args.place_adv[l] == 'Y':

					# reshape the features (e.g. 128x5x2 --> 640x2)
					pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
					pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

					# prepare domain labels
					source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
					target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
					domain_label = torch.cat((source_domain_label,target_domain_label),0)

					domain_label = domain_label.cuda(non_blocking=True)

					pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
					pred_domain_all.append(pred_domain)
					pred_domain_target_all.append(pred_domain_target_single)

					if args.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
						pred_domain = pred_domain / pred_domain.var().log()
					loss_adversarial_single = criterion_domain(pred_domain, domain_label)

					loss_adversarial += loss_adversarial_single

			losses_a.update(loss_adversarial.item(), pred_domain.size(0))
			loss += loss_adversarial

		# (III) other loss
		# 1. entropy loss for target data
		if args.add_loss_DA == 'target_entropy' and args.use_target != 'none':
			loss_entropy = cross_entropy_soft(out_target)
			losses_e.update(loss_entropy.item(), out_target.size(0))
			loss += gamma * loss_entropy

		# 2. discrepancy loss for MCD (CVPR 18)
		if args.ens_DA == 'MCD' and args.use_target != 'none':
			_, _, _, _, _, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=True)

			# ignore dummy tensors
			_, out_target, out_target_2, _, _ = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)

			loss_dis = -dis_MCD(out_target, out_target_2)
			losses_s.update(loss_dis.item(), out_target.size(0))
			loss += loss_dis

		# 3. attentive entropy loss
		if args.add_loss_DA == 'attentive_entropy' and args.use_attn != 'none' and args.use_target != 'none':
			loss_entropy = attentive_entropy(torch.cat((out_source, out_target),0), pred_domain_all[1])
			losses_e.update(loss_entropy.item(), out_target.size(0))
			loss += gamma * loss_entropy

		# measure accuracy and record loss
		pred = out

		prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))

		losses.update(loss.item())
		top1.update(prec1.item(), out_source.size(0))
		top5.update(prec5.item(), out_source.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()

		loss.backward()

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient and args.verbose:
				print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
				   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
				   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
				   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
				   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' + \
				   'Loss {loss.val:.4f} ({loss.avg:.4f})   loss_c {loss_c.avg:.4f}\t'

			if args.dis_DA != 'none' and args.use_target != 'none':
				line += 'alpha {alpha:.3f}  loss_d {loss_d.avg:.4f}\t'

			if args.adv_DA != 'none' and args.use_target != 'none':
				line += 'beta {beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}  loss_a {loss_a.avg:.4f}\t'

			if args.add_loss_DA != 'none' and args.use_target != 'none':
				line += 'gamma {gamma:.6f}  loss_e {loss_e.avg:.4f}\t'

			if args.ens_DA != 'none' and args.use_target != 'none':
				line += 'mu {mu:.6f}  loss_s {loss_s.avg:.4f}\t'

			line = line.format(
				epoch, i, len(source_loader), batch_time=batch_time, data_time=data_time, alpha=alpha, beta=beta, gamma=gamma, mu=mu,
				loss=losses, loss_c=losses_c, loss_d=losses_d, loss_a=losses_a, loss_e=losses_e, loss_s=losses_s, top1=top1, top5=top5,
				lr=optimizer.param_groups[0]['lr'])

			if i % args.show_freq == 0:
				print(line)

			log.write('%s\n' % line)

		# adjust the learning rate for ech step (e.g. DANN)
		if args.lr_adaptive == 'dann':
			adjust_learning_rate_dann(optimizer, p)

		# save attention values w/ the selected class
		if args.save_attention >= 0:
			attn_source = attn_source[source_label==args.save_attention]
			attn_target = attn_target[target_label==args.save_attention]
			attn_epoch_source = torch.cat((attn_epoch_source, attn_source.cpu()))
			attn_epoch_target = torch.cat((attn_epoch_target, attn_target.cpu()))

	# update the embedding every epoch
	if args.tensorboard:
		n_iter_train = epoch * len(source_loader) # calculate the total iteration
		# embedding
		# see source and target separately
		writer.add_embedding(feat_source_display, metadata=label_source_display.data, global_step=n_iter_train, tag='train_source')
		writer.add_embedding(feat_target_display, metadata=label_target_display.data, global_step=n_iter_train, tag='train_target')

		# mix source and target
		feat_all_display = torch.cat((feat_source_display, feat_target_display), 0)
		label_all_domain_display = torch.cat((label_source_domain_display, label_target_domain_display), 0)
		writer.add_embedding(feat_all_display, metadata=label_all_domain_display.data, global_step=n_iter_train, tag='train_DA')

		# emphazise some classes (1, 3, 11 here)
		label_source_1 = 1 * torch.eq(label_source_display, torch.cuda.LongTensor([1]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
		label_source_3 = 2 * torch.eq(label_source_display, torch.cuda.LongTensor([3]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
		label_source_11 = 3 * torch.eq(label_source_display, torch.cuda.LongTensor([11]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)

		label_target_1 = 4 * torch.eq(label_target_display, torch.cuda.LongTensor([1]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
		label_target_3 = 5 * torch.eq(label_target_display, torch.cuda.LongTensor([3]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
		label_target_11 = 6 * torch.eq(label_target_display, torch.cuda.LongTensor([11]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)

		label_source_display_new = label_source_1 + label_source_3 + label_source_11
		id_source_show = ~torch.eq(label_source_display_new, 0).cuda(non_blocking=True)
		label_source_display_new = label_source_display_new[id_source_show]
		feat_source_display_new = feat_source_display[id_source_show]

		label_target_display_new = label_target_1 + label_target_3 + label_target_11
		id_target_show = ~torch.eq(label_target_display_new, 0).cuda(non_blocking=True)
		label_target_display_new = label_target_display_new[id_target_show]
		feat_target_display_new = feat_target_display[id_target_show]

		feat_all_display_new = torch.cat((feat_source_display_new, feat_target_display_new), 0)
		label_all_display_new = torch.cat((label_source_display_new, label_target_display_new), 0)
		writer.add_embedding(feat_all_display_new, metadata=label_all_display_new.data, global_step=n_iter_train, tag='train_DA_labels')

	log_short.write('%s\n' % line)
	return losses_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0)

def validate(val_loader, model, criterion, num_class, epoch, log):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()

	# initialize the embedding
	if args.tensorboard:
		feat_val_display = None
		label_val_display = None

	for i, (val_data, val_label) in enumerate(val_loader):

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < args.batch_size[2]:
			val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data = torch.cat((val_data, val_data_dummy))

		val_label = val_label.cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_frame = val_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames

			# compute output
			_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(val_data, val_data, [0]*len(args.beta), 0, is_train=False, reverse=False)

			# ignore dummy tensors
			attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori)

			# measure accuracy and record loss
			label = val_label_frame if args.baseline_type == 'frame' else val_label

			# store the embedding
			if args.tensorboard:
				feat_val_display = feat_val[1] if i == 0 else torch.cat((feat_val_display, feat_val[1]), 0)
				label_val_display = label if i == 0 else torch.cat((label_val_display, label), 0)

			pred = out_val

			if args.baseline_type == 'tsn':
				pred = pred.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss = criterion(pred, label)
			prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))

			losses.update(loss.item(), out_val.size(0))
			top1.update(prec1.item(), out_val.size(0))
			top5.update(prec5.item(), out_val.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				line = 'Test: [{0}][{1}/{2}]\t' + \
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'

				line = line.format(
					   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5)

				if i % args.show_freq == 0:
					print(line)

				log.write('%s\n' % line)

	if args.tensorboard:  # update the embedding every iteration
		# embedding
		n_iter_val = epoch * len(val_loader)

		writer.add_embedding(feat_val_display, metadata=label_val_display.data, global_step=n_iter_val, tag='validation')

	print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses)))

	return top1.avg


def save_checkpoint(state, is_best, path_exp, filename='checkpoint.pth.tar'):

	path_file = path_exp + filename
	torch.save(state, path_file)
	if is_best:
		path_best = path_exp + 'model_best.pth.tar'
		shutil.copyfile(path_file, path_best)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, decay):
	"""Sets the learning rate to the initial LR decayed by 10 """
	for param_group in optimizer.param_groups:
		param_group['lr'] /= decay

def adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
	ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
	if ops[op](stat_current, stat_previous):
		for param_group in optimizer.param_groups:
			param_group['lr'] /= decay

def adjust_learning_rate_dann(optimizer, p):
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr / (1. + 10 * p) ** 0.75

def loss_adaptive_weight(loss, pred):
	weight = 1 / pred.var().log()
	constant = pred.std().log()
	return loss * weight + constant

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

# remove dummy tensors
def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

if __name__ == '__main__':
	main()
