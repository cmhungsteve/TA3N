from torch import nn

from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import TRNmodule
import math

from colorama import init
from colorama import Fore, Back, Style

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

# definition of Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None

# definition of Gradient Scaling Layer
class GradScale(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output * ctx.beta
		return grad_input, None

# definition of Temporal-ConvNet Layer
class TCL(nn.Module):
	def __init__(self, conv_size, dim):
		super(TCL, self).__init__()

		self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size,1), padding=(conv_size//2,0))

		# initialization
		kaiming_normal_(self.conv2d.weight)

	def	forward(self, x):
		x = self.conv2d(x)

		return x

class VideoModel(nn.Module):
	def __init__(self, num_class, baseline_type, frame_aggregation, modality,
				train_segments=5, val_segments=25,
				base_model='resnet101', path_pretrained='', new_length=None,
				before_softmax=True,
				dropout_i=0.5, dropout_v=0.5, use_bn='none', ens_DA='none',
				crop_num=1, partial_bn=True, verbose=True, add_fc=1, fc_dim=1024,
				n_rnn=1, rnn_cell='LSTM', n_directions=1, n_ts=5,
				use_attn='TransAttn', n_attn=1, use_attn_frame='none',
				share_params='Y'):
		super(VideoModel, self).__init__()
		self.modality = modality
		self.train_segments = train_segments
		self.val_segments = val_segments
		self.baseline_type = baseline_type
		self.frame_aggregation = frame_aggregation
		self.reshape = True
		self.before_softmax = before_softmax
		self.dropout_rate_i = dropout_i
		self.dropout_rate_v = dropout_v
		self.use_bn = use_bn
		self.ens_DA = ens_DA
		self.crop_num = crop_num
		self.add_fc = add_fc
		self.fc_dim = fc_dim
		self.share_params = share_params

		# RNN
		self.n_layers = n_rnn
		self.rnn_cell = rnn_cell
		self.n_directions = n_directions
		self.n_ts = n_ts # temporal segment

		# Attention
		self.use_attn = use_attn 
		self.n_attn = n_attn
		self.use_attn_frame = use_attn_frame

		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		if verbose:
			print(("""
				Initializing TSN with base model: {}.
				TSN Configurations:
				input_modality:     {}
				num_segments:       {}
				new_length:         {}
				""".format(base_model, self.modality, self.train_segments, self.new_length)))

		self._prepare_DA(num_class, base_model)

		if not self.before_softmax:
			self.softmax = nn.Softmax()

		self._enable_pbn = partial_bn
		if partial_bn:
			self.partialBN(True)

	def _prepare_DA(self, num_class, base_model): # convert the model to DA framework
		if base_model == 'c3d': # C3D mode: in construction...
			from C3D_model import C3D
			model_test = C3D()
			self.feature_dim = model_test.fc7.in_features
		else:
			model_test = getattr(torchvision.models, base_model)(True) # model_test is only used for getting the dim #
			self.feature_dim = model_test.fc.in_features

		std = 0.001
		feat_shared_dim = min(self.fc_dim, self.feature_dim) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
		feat_frame_dim = feat_shared_dim

		self.relu = nn.ReLU(inplace=True)
		self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
		self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

		#------ frame-level layers (shared layers + source layers + domain layers) ------#
		if self.add_fc < 1:
			raise ValueError(Back.RED + 'add at least one fc layer')

		# 1. shared feature layers
		self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
		normal_(self.fc_feature_shared_source.weight, 0, std)
		constant_(self.fc_feature_shared_source.bias, 0)

		if self.add_fc > 1:
			self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_2_source.weight, 0, std)
			constant_(self.fc_feature_shared_2_source.bias, 0)

		if self.add_fc > 2:
			self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_3_source.weight, 0, std)
			constant_(self.fc_feature_shared_3_source.bias, 0)

		# 2. frame-level feature layers
		self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
		normal_(self.fc_feature_source.weight, 0, std)
		constant_(self.fc_feature_source.bias, 0)

		# 3. domain feature layers (frame-level)
		self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
		normal_(self.fc_feature_domain.weight, 0, std)
		constant_(self.fc_feature_domain.bias, 0)

		# 4. classifiers (frame-level)
		self.fc_classifier_source = nn.Linear(feat_frame_dim, num_class)
		normal_(self.fc_classifier_source.weight, 0, std)
		constant_(self.fc_classifier_source.bias, 0)

		self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
		normal_(self.fc_classifier_domain.weight, 0, std)
		constant_(self.fc_classifier_domain.bias, 0)

		if self.share_params == 'N':
			self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_target.weight, 0, std)
			constant_(self.fc_feature_shared_target.bias, 0)
			if self.add_fc > 1:
				self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_2_target.weight, 0, std)
				constant_(self.fc_feature_shared_2_target.bias, 0)
			if self.add_fc > 2:
				self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_3_target.weight, 0, std)
				constant_(self.fc_feature_shared_3_target.bias, 0)

			self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
			normal_(self.fc_feature_target.weight, 0, std)
			constant_(self.fc_feature_target.bias, 0)
			self.fc_classifier_target = nn.Linear(feat_frame_dim, num_class)
			normal_(self.fc_classifier_target.weight, 0, std)
			constant_(self.fc_classifier_target.bias, 0)

		# BN for the above layers
		if self.use_bn != 'none':  # S & T: use AdaBN (ICLRW 2017) approach
			self.bn_shared_S = nn.BatchNorm1d(feat_shared_dim)  # BN for the shared layers
			self.bn_shared_T = nn.BatchNorm1d(feat_shared_dim)
			self.bn_source_S = nn.BatchNorm1d(feat_frame_dim)  # BN for the source feature layers
			self.bn_source_T = nn.BatchNorm1d(feat_frame_dim)

		#------ aggregate frame-based features (frame feature --> video feature) ------#
		if self.frame_aggregation == 'rnn': # 2. rnn
			self.hidden_dim = feat_frame_dim
			if self.rnn_cell == 'LSTM':
				self.rnn = nn.LSTM(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))
			elif self.rnn_cell == 'GRU':
				self.rnn = nn.GRU(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))

			# initialization
			for p in range(self.n_layers):
				kaiming_normal_(self.rnn.all_weights[p][0])
				kaiming_normal_(self.rnn.all_weights[p][1])

			self.bn_before_rnn = nn.BatchNorm2d(1)
			self.bn_after_rnn = nn.BatchNorm2d(1)

		elif self.frame_aggregation == 'trn': # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 512
			self.TRN = TRNmodule.RelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		elif self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 256
			self.TRN = TRNmodule.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

		elif self.frame_aggregation == 'temconv': # 3. temconv

			self.tcl_3_1 = TCL(3, 1)
			self.tcl_5_1 = TCL(5, 1)
			self.bn_1_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_1_T = nn.BatchNorm1d(feat_frame_dim)

			self.tcl_3_2 = TCL(3, 1)
			self.tcl_5_2 = TCL(5, 2)
			self.bn_2_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_2_T = nn.BatchNorm1d(feat_frame_dim)

			self.conv_fusion = nn.Sequential(
				nn.Conv2d(2, 1, kernel_size=(1, 1), padding=(0, 0)),
				nn.ReLU(inplace=True),
			)

		# ------ video-level layers (source layers + domain layers) ------#
		if self.frame_aggregation == 'avgpool': # 1. avgpool
			feat_aggregated_dim = feat_shared_dim
		if 'trn' in self.frame_aggregation : # 4. trn
			feat_aggregated_dim = self.num_bottleneck
		elif self.frame_aggregation == 'rnn': # 2. rnn
			feat_aggregated_dim = self.hidden_dim
		elif self.frame_aggregation == 'temconv': # 3. temconv
			feat_aggregated_dim = feat_shared_dim

		feat_video_dim = feat_aggregated_dim

		# 1. source feature layers (video-level)
		self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_video_source.weight, 0, std)
		constant_(self.fc_feature_video_source.bias, 0)

		self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
		normal_(self.fc_feature_video_source_2.weight, 0, std)
		constant_(self.fc_feature_video_source_2.bias, 0)

		# 2. domain feature layers (video-level)
		self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_domain_video.weight, 0, std)
		constant_(self.fc_feature_domain_video.bias, 0)

		# 3. classifiers (video-level)
		self.fc_classifier_video_source = nn.Linear(feat_video_dim, num_class)
		normal_(self.fc_classifier_video_source.weight, 0, std)
		constant_(self.fc_classifier_video_source.bias, 0)

		if self.ens_DA == 'MCD':
			self.fc_classifier_video_source_2 = nn.Linear(feat_video_dim, num_class) # second classifier for self-ensembling
			normal_(self.fc_classifier_video_source_2.weight, 0, std)
			constant_(self.fc_classifier_video_source_2.bias, 0)

		self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
		normal_(self.fc_classifier_domain_video.weight, 0, std)
		constant_(self.fc_classifier_domain_video.bias, 0)

		# domain classifier for TRN-M
		if self.frame_aggregation == 'trn-m':
			self.relation_domain_classifier_all = nn.ModuleList()
			for i in range(self.train_segments-1):
				relation_domain_classifier = nn.Sequential(
					nn.Linear(feat_aggregated_dim, feat_video_dim),
					nn.ReLU(),
					nn.Linear(feat_video_dim, 2)
				)
				self.relation_domain_classifier_all += [relation_domain_classifier]

		if self.share_params == 'N':
			self.fc_feature_video_target = nn.Linear(feat_aggregated_dim, feat_video_dim)
			normal_(self.fc_feature_video_target.weight, 0, std)
			constant_(self.fc_feature_video_target.bias, 0)
			self.fc_feature_video_target_2 = nn.Linear(feat_video_dim, feat_video_dim)
			normal_(self.fc_feature_video_target_2.weight, 0, std)
			constant_(self.fc_feature_video_target_2.bias, 0)
			self.fc_classifier_video_target = nn.Linear(feat_video_dim, num_class)
			normal_(self.fc_classifier_video_target.weight, 0, std)
			constant_(self.fc_classifier_video_target.bias, 0)

		# BN for the above layers
		if self.use_bn != 'none':  # S & T: use AdaBN (ICLRW 2017) approach
			self.bn_source_video_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_T = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_T = nn.BatchNorm1d(feat_video_dim)

		self.alpha = torch.ones(1)
		if self.use_bn == 'AutoDIAL':
			self.alpha = nn.Parameter(self.alpha)

		# ------ attention mechanism ------#
		# conventional attention
		if self.use_attn == 'general':
			self.attn_layer = nn.Sequential(
				nn.Linear(feat_aggregated_dim, feat_aggregated_dim),
				nn.Tanh(),
				nn.Linear(feat_aggregated_dim, 1)
				)


	def train(self, mode=True):
		# not necessary in our setting
		"""
		Override the default train() to freeze the BN parameters
		:return:
		"""
		super(VideoModel, self).train(mode)
		count = 0
		if self._enable_pbn:
			print("Freezing BatchNorm2D except the first one.")
			for m in self.base_model.modules():
				if isinstance(m, nn.BatchNorm2d):
					count += 1
					if count >= (2 if self._enable_pbn else 1):
						m.eval()

						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False

	def partialBN(self, enable):
		self._enable_pbn = enable

	def get_trans_attn(self, pred_domain):
		softmax = nn.Softmax(dim=1)
		logsoftmax = nn.LogSoftmax(dim=1)
		entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
		weights = 1 - entropy

		return weights

	def get_general_attn(self, feat):
		num_segments = feat.size()[1]
		feat = feat.view(-1, feat.size()[-1]) # reshape features: 128x4x256 --> (128x4)x256
		weights = self.attn_layer(feat) # e.g. (128x4)x1
		weights = weights.view(-1, num_segments, weights.size()[-1]) # reshape attention weights: (128x4)x1 --> 128x4x1
		weights = F.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

		return weights

	def get_attn_feat_frame(self, feat_fc, pred_domain): # not used for now
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, 1).repeat(1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 512)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn

	def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn, weights_attn[:,:,0]

	def aggregate_frames(self, feat_fc, num_segments, pred_domain):
		feat_fc_video = None
		if self.frame_aggregation == 'rnn':
			# 2. RNN
			feat_fc_video = feat_fc.view((-1, num_segments) + feat_fc.size()[-1:])  # reshape for RNN

			# temporal segments and pooling
			len_ts = round(num_segments/self.n_ts)
			num_extra_f = len_ts*self.n_ts-num_segments
			if num_extra_f < 0: # can remove last frame-level features
				feat_fc_video = feat_fc_video[:, :len_ts * self.n_ts, :]  # make the temporal length can be divided by n_ts (16 x 25 x 512 --> 16 x 24 x 512)
			elif num_extra_f > 0: # need to repeat last frame-level features
				feat_fc_video = torch.cat((feat_fc_video, feat_fc_video[:,-1:,:].repeat(1,num_extra_f,1)), 1) # make the temporal length can be divided by n_ts (16 x 5 x 512 --> 16 x 6 x 512)

			feat_fc_video = feat_fc_video.view(
				(-1, self.n_ts, len_ts) + feat_fc_video.size()[2:])  # 16 x 6 x 512 --> 16 x 3 x 2 x 512
			feat_fc_video = nn.MaxPool2d(kernel_size=(len_ts, 1))(
				feat_fc_video)  # 16 x 3 x 2 x 512 --> 16 x 3 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(2)  # 16 x 3 x 1 x 512 --> 16 x 3 x 512

			hidden_temp = torch.zeros(self.n_layers * self.n_directions, feat_fc_video.size(0),
									  self.hidden_dim // self.n_directions).cuda()

			if self.rnn_cell == 'LSTM':
				hidden_init = (hidden_temp, hidden_temp)
			elif self.rnn_cell == 'GRU':
				hidden_init = hidden_temp

			self.rnn.flatten_parameters()
			feat_fc_video, hidden_final = self.rnn(feat_fc_video, hidden_init)  # e.g. 16 x 25 x 512

			# get the last feature vector
			feat_fc_video = feat_fc_video[:, -1, :]

		else:
			# 1. averaging
			feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
			if self.use_attn == 'TransAttn': # get the attention weighting
				weights_attn = self.get_trans_attn(pred_domain)
				weights_attn = weights_attn.view(-1, 1, num_segments,1).repeat(1,1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
				feat_fc_video = (weights_attn+1) * feat_fc_video

			feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

		return feat_fc_video

	def final_output(self, pred, pred_video, num_segments):
		if self.baseline_type == 'video':
			base_out = pred_video
		else:
			base_out = pred

		if not self.before_softmax:
			base_out = self.softmax(base_out)

		output = base_out

		if self.baseline_type == 'tsn':
			if self.reshape:
				base_out = base_out.view((-1, num_segments) + base_out.size()[1:]) # e.g. 16 x 3 x 12 (3 segments)

			output = base_out.mean(1) # e.g. 16 x 12

		return output

	def domain_classifier_frame(self, feat, beta):
		feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
		feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
		feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
		pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

		return pred_fc_domain_frame

	def domain_classifier_video(self, feat_video, beta):
		feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
		feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
		feat_fc_domain_video = self.relu(feat_fc_domain_video)
		pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

		return pred_fc_domain_video

	def domain_classifier_relation(self, feat_relation, beta):
		# 128x4x256 --> (128x4)x2
		pred_fc_domain_relation_video = None
		for i in range(len(self.relation_domain_classifier_all)):
			feat_relation_single = feat_relation[:,i,:].squeeze(1) # 128x1x256 --> 128x256
			feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[0]) # the same beta for all relations (for now)

			pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
	
			if pred_fc_domain_relation_video is None:
				pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
			else:
				pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
		
		pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)

		return pred_fc_domain_relation_video

	def domainAlign(self, input_S, input_T, is_train, name_layer, alpha, num_segments, dim):
		input_S = input_S.view((-1, dim, num_segments) + input_S.size()[-1:])  # reshape based on the segments (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
		input_T = input_T.view((-1, dim, num_segments) + input_T.size()[-1:])  # reshape based on the segments

		# clamp alpha
		alpha = max(alpha,0.5)

		# rearange source and target data
		num_S_1 = int(round(input_S.size(0) * alpha))
		num_S_2 = input_S.size(0) - num_S_1
		num_T_1 = int(round(input_T.size(0) * alpha))
		num_T_2 = input_T.size(0) - num_T_1

		if is_train and num_S_2 > 0 and num_T_2 > 0:
			input_source = torch.cat((input_S[:num_S_1], input_T[-num_T_2:]), 0)
			input_target = torch.cat((input_T[:num_T_1], input_S[-num_S_2:]), 0)
		else:
			input_source = input_S
			input_target = input_T

		# adaptive BN
		input_source = input_source.view((-1, ) + input_source.size()[-1:]) # reshape to feed BN (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
		input_target = input_target.view((-1, ) + input_target.size()[-1:])

		if name_layer == 'shared':
			input_source_bn = self.bn_shared_S(input_source)
			input_target_bn = self.bn_shared_T(input_target)
		elif 'trn' in name_layer:
			input_source_bn = self.bn_trn_S(input_source)
			input_target_bn = self.bn_trn_T(input_target)
		elif name_layer == 'temconv_1':
			input_source_bn = self.bn_1_S(input_source)
			input_target_bn = self.bn_1_T(input_target)
		elif name_layer == 'temconv_2':
			input_source_bn = self.bn_2_S(input_source)
			input_target_bn = self.bn_2_T(input_target)

		input_source_bn = input_source_bn.view((-1, dim, num_segments) + input_source_bn.size()[-1:])  # reshape back (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
		input_target_bn = input_target_bn.view((-1, dim, num_segments) + input_target_bn.size()[-1:])  #

		# rearange back to the original order of source and target data (since target may be unlabeled)
		if is_train and num_S_2 > 0 and num_T_2 > 0:
			input_source_bn = torch.cat((input_source_bn[:num_S_1], input_target_bn[-num_S_2:]), 0)
			input_target_bn = torch.cat((input_target_bn[:num_T_1], input_source_bn[-num_T_2:]), 0)

		# reshape for frame-level features
		if name_layer == 'shared' or name_layer == 'trn_sum':
			input_source_bn = input_source_bn.view((-1,) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
			input_target_bn = input_target_bn.view((-1,) + input_target_bn.size()[-1:])
		elif name_layer == 'trn':
			input_source_bn = input_source_bn.view((-1, num_segments) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
			input_target_bn = input_target_bn.view((-1, num_segments) + input_target_bn.size()[-1:])

		return input_source_bn, input_target_bn

	def forward(self, input_source, input_target, beta, mu, is_train, reverse):
		batch_source = input_source.size()[0]
		batch_target = input_target.size()[0]
		num_segments = self.train_segments if is_train else self.val_segments
		# sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
		sample_len = self.new_length
		feat_all_source = []
		feat_all_target = []
		pred_domain_all_source = []
		pred_domain_all_target = []

		# input_data is a list of tensors --> need to do pre-processing
		feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 256 x 25 x 2048 --> 6400 x 2048
		feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048

		#=== shared layers ===#
		# need to separate BN for source & target ==> otherwise easy to overfit to source data
		if self.add_fc < 1:
			raise ValueError(Back.RED + 'not enough fc layer')

		feat_fc_source = self.fc_feature_shared_source(feat_base_source)
		feat_fc_target = self.fc_feature_shared_target(feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)

		# adaptive BN
		if self.use_bn != 'none':
			feat_fc_source, feat_fc_target = self.domainAlign(feat_fc_source, feat_fc_target, is_train, 'shared', self.alpha.item(), num_segments, 1)

		feat_fc_source = self.relu(feat_fc_source)
		feat_fc_target = self.relu(feat_fc_target)
		feat_fc_source = self.dropout_i(feat_fc_source)
		feat_fc_target = self.dropout_i(feat_fc_target)

		# feat_fc = self.dropout_i(feat_fc)
		feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
		feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		if self.add_fc > 1: 
			feat_fc_source = self.fc_feature_shared_2_source(feat_fc_source)
			feat_fc_target = self.fc_feature_shared_2_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_2_source(feat_fc_target)

			feat_fc_source = self.relu(feat_fc_source)
			feat_fc_target = self.relu(feat_fc_target)
			feat_fc_source = self.dropout_i(feat_fc_source)
			feat_fc_target = self.dropout_i(feat_fc_target)

			feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		if self.add_fc > 2: 
			feat_fc_source = self.fc_feature_shared_3_source(feat_fc_source)
			feat_fc_target = self.fc_feature_shared_3_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_3_source(feat_fc_target)

			feat_fc_source = self.relu(feat_fc_source)
			feat_fc_target = self.relu(feat_fc_target)
			feat_fc_source = self.dropout_i(feat_fc_source)
			feat_fc_target = self.dropout_i(feat_fc_target)

			feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		# === adversarial branch (frame-level) ===#
		pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)
		pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)

		pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

		if self.use_attn_frame != 'none': # attend the frame-level features only
			feat_fc_source = self.get_attn_feat_frame(feat_fc_source, pred_fc_domain_frame_source)
			feat_fc_target = self.get_attn_feat_frame(feat_fc_target, pred_fc_domain_frame_target)

		#=== source layers (frame-level) ===#
		pred_fc_source = self.fc_classifier_source(feat_fc_source)
		pred_fc_target = self.fc_classifier_target(feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source(feat_fc_target)
		if self.baseline_type == 'frame':
			feat_all_source.append(pred_fc_source.view((batch_source, num_segments) + pred_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(pred_fc_target.view((batch_target, num_segments) + pred_fc_target.size()[-1:]))

		### aggregate the frame-based features to video-based features ###
		if self.frame_aggregation == 'avgpool' or self.frame_aggregation == 'rnn':
			feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source)
			feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

			attn_relation_source = feat_fc_video_source[:,0] # assign random tensors to attention values to avoid runtime error
			attn_relation_target = feat_fc_video_target[:,0] # assign random tensors to attention values to avoid runtime error

		elif 'trn' in self.frame_aggregation:
			feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
			feat_fc_video_relation_target = self.TRN(feat_fc_video_target)

			# adversarial branch
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

			# transferable attention
			if self.use_attn != 'none': # get the attention weighting
				feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_segments)
				feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_segments)
			else:
				attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
				attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error

			# sum up relation features (ignore 1-relation)
			feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
			feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)

		elif self.frame_aggregation == 'temconv': # DA operation inside temconv
			feat_fc_video_source = feat_fc_source.view((-1, 1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments
			feat_fc_video_target = feat_fc_target.view((-1, 1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments

			# 1st TCL
			feat_fc_video_source_3_1 = self.tcl_3_1(feat_fc_video_source)
			feat_fc_video_target_3_1 = self.tcl_3_1(feat_fc_video_target)

			if self.use_bn != 'none':
				feat_fc_video_source_3_1, feat_fc_video_target_3_1 = self.domainAlign(feat_fc_video_source_3_1, feat_fc_video_target_3_1, is_train, 'temconv_1', self.alpha.item(), num_segments, 1)

			feat_fc_video_source = self.relu(feat_fc_video_source_3_1)  # 16 x 1 x 5 x 512
			feat_fc_video_target = self.relu(feat_fc_video_target_3_1)  # 16 x 1 x 5 x 512

			feat_fc_video_source = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_source)  # 16 x 4 x 1 x 512
			feat_fc_video_target = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_target)  # 16 x 4 x 1 x 512

			feat_fc_video_source = feat_fc_video_source.squeeze(1).squeeze(1)  # e.g. 16 x 512
			feat_fc_video_target = feat_fc_video_target.squeeze(1).squeeze(1)  # e.g. 16 x 512

		if self.baseline_type == 'video':
			feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
			feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))

		#=== source layers (video-level) ===#
		feat_fc_video_source = self.dropout_v(feat_fc_video_source)
		feat_fc_video_target = self.dropout_v(feat_fc_video_target)

		if reverse:
			feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
			feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

		pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
		pred_fc_video_target = self.fc_classifier_video_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)

		if self.baseline_type == 'video': # only store the prediction from classifier 1 (for now)
			feat_all_source.append(pred_fc_video_source.view((batch_source,) + pred_fc_video_source.size()[-1:]))
			feat_all_target.append(pred_fc_video_target.view((batch_target,) + pred_fc_video_target.size()[-1:]))

		#=== adversarial branch (video-level) ===#
		pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
		pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)

		pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

		# video relation-based discriminator
		if self.frame_aggregation == 'trn-m':
			num_relation = feat_fc_video_relation_source.size()[1]
			pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
			pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
		else:
			pred_domain_all_source.append(pred_fc_domain_video_source) # if not trn-m, add dummy tensors for relation features
			pred_domain_all_target.append(pred_fc_domain_video_target)

		#=== final output ===#
		output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments) # select output from frame or video prediction
		output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

		output_source_2 = output_source
		output_target_2 = output_target

		if self.ens_DA == 'MCD':
			pred_fc_video_source_2 = self.fc_classifier_video_source_2(feat_fc_video_source)
			pred_fc_video_target_2 = self.fc_classifier_video_target_2(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source_2(feat_fc_video_target)
			output_source_2 = self.final_output(pred_fc_source, pred_fc_video_source_2, num_segments)
			output_target_2 = self.final_output(pred_fc_target, pred_fc_video_target_2, num_segments)

		return attn_relation_source, output_source, output_source_2, pred_domain_all_source[::-1], feat_all_source[::-1], attn_relation_target, output_target, output_target_2, pred_domain_all_target[::-1], feat_all_target[::-1] # reverse the order of feature list due to some multi-gpu issues
