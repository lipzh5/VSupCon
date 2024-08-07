from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import transformers
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import os.path as osp
import logging
log = logging.getLogger(__name__)
try:
	import apex
	from apex import amp, optimizers
except ImportError:
	pass



def parse_option():
	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--print_freq', type=int, default=200,
						help='print frequency')
	parser.add_argument('--save_freq', type=int, default=50,
						help='save frequency')
	parser.add_argument('--batch_size', type=int, default=1024, # 256,
						help='batch_size')
	parser.add_argument('--num_workers', type=int, default=16,
						help='num of workers to use')
	parser.add_argument('--epochs', type=int, default=100, # 1000,
						help='number of training epochs')

	# optimization
	parser.add_argument('--learning_rate', type=float, default=0.0001, # 0.05 for resnet50; 0.001 for inceptionresnetv1
						help='learning rate')
	parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
						help='where to decay lr, can be a list')
	parser.add_argument('--lr_decay_rate', type=float, default=0.1,
						help='decay rate for learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4,
						help='weight decay')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='momentum')

	# model dataset
	parser.add_argument('--model', type=str, default='inceptionresnetv1')  # resnet50
	parser.add_argument('--dataset', type=str, default='affwild2',  # cifar10
						choices=['cifar10', 'cifar100', 'path', 'affwild2'], help='dataset')
	parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
	parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
	parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
	parser.add_argument('--size', type=int, default=160, help='parameter for RandomResizedCrop')  # 32

	# method
	parser.add_argument('--method', type=str, default='SupCon',
						choices=['SupCon', 'SimCLR'], help='choose method')

	# temperature
	parser.add_argument('--temp', type=float, default=0.07,
						help='temperature for loss function')

	# other setting
	parser.add_argument('--cosine', action='store_true', default=True,
						help='using cosine annealing')
	parser.add_argument('--syncBN', action='store_true',
						help='using synchronized batch normalization')
	parser.add_argument('--warm', action='store_true',
						help='warm-up for large batch training')
	parser.add_argument('--trial', type=str, default='0',
						help='id for recording multiple runs')
	parser.add_argument('--device_id', type=int, default=0)
	parser.add_argument('--data_dsr', type=int, default=1, help='data set downsampling ratio, e.g., 1 stands for original, 2 stands for original//2')
	parser.add_argument('--warmup_from', type=float, default=1e-3)
	parser.add_argument('--warm_epochs', type=int, default=10)
	parser.add_argument('--dflr', action='store_true', default=False, help='whether to use different lr for encoder and head')
	parser.add_argument('--use_webface_pretrain', action='store_true')
	parser.add_argument('--weight_init', action='store_true', help='using customized weight init')

	opt = parser.parse_args()

	# check if dataset is path that passed required arguments
	if opt.dataset == 'path':
		assert opt.data_folder is not None \
			and opt.mean is not None \
			and opt.std is not None

	# set the path according to the environment
	if opt.data_folder is None:
		opt.data_folder = './datasets/'
	opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
	opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

	iterations = opt.lr_decay_epochs.split(',')
	opt.lr_decay_epochs = list([])
	for it in iterations:
		opt.lr_decay_epochs.append(int(it))
	
	# device_ids = opt.device_ids.split(',')
	# opt.device_ids = [int(it) for it in device_ids]

	opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
		format(opt.method, opt.dataset, opt.model, opt.learning_rate,
			   opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

	if opt.cosine:
		opt.model_name = '{}_cosine'.format(opt.model_name)

	# warm-up for large-batch training,
	if opt.batch_size > 256:
		opt.warm = True
	if opt.warm:
		opt.model_name = '{}_warm'.format(opt.model_name)
		# opt.warmup_from = 0.01
		# opt.warm_epochs = 10
		if opt.cosine:
			eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
			opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
					1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
		else:
			opt.warmup_to = opt.learning_rate

	opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
	if not os.path.isdir(opt.tb_folder):
		os.makedirs(opt.tb_folder)

	opt.save_folder = os.path.join(opt.model_path, opt.model_name)
	if not os.path.isdir(opt.save_folder):
		os.makedirs(opt.save_folder)

	return opt

from datasets.affwild2 import get_affwild2_dataset

def set_loader(opt):
	# construct data loader
	if opt.dataset == 'cifar10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2023, 0.1994, 0.2010)
	elif opt.dataset == 'cifar100':
		mean = (0.5071, 0.4867, 0.4408)
		std = (0.2675, 0.2565, 0.2761)
	elif opt.dataset == 'affwild2':  
		if opt.model == 'inceptionresnetv1': 
			'''masked cropped image'''
			mean =  (0.0722, 0.0489, 0.0451)
			std = (0.1807, 0.1274, 0.1169)

			# mean = (0.4654, 0.3532, 0.3217)
			# std = (0.2334, 0.2003, 0.1902)
			opt.size = 160
		else:  # original size 
			mean = (0.4652, 0.3531, 0.3215)
			std = (0.2348, 0.2019, 0.1918)
		# if opt.model in {'resnet50', }:
		#     assert opt.size == 112, "picture size must be 112 for affwild2"
	elif opt.dataset == 'path':
		mean = eval(opt.mean)
		std = eval(opt.std)
	else:
		raise ValueError('dataset not supported: {}'.format(opt.dataset))
	normalize = transforms.Normalize(mean=mean, std=std)

	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		normalize,
	])

	trans_list = [transforms.Resize(size=opt.size),] if opt.model=='inceptionresnetv1' else [] 
	val_transform = transforms.Compose(
		trans_list + [
		transforms.ToTensor(),
		normalize,
	])

	if opt.dataset == 'cifar10':
		train_dataset = datasets.CIFAR10(root=opt.data_folder,
										 transform=TwoCropTransform(train_transform),
										 download=True)

		val_dataset = datasets.CIFAR10(root=opt.data_folder,
									   train=False,
									   transform=TwoCropTransform(val_transform))  
	elif opt.dataset == 'affwild2':
		train_dataset = get_affwild2_dataset('train', TwoCropTransform(train_transform), opt.data_dsr)
		val_dataset = get_affwild2_dataset('val', TwoCropTransform(val_transform), opt.data_dsr)
	elif opt.dataset == 'cifar100':
		train_dataset = datasets.CIFAR100(root=opt.data_folder,
										  transform=TwoCropTransform(train_transform),
										  download=True)
	elif opt.dataset == 'path':
		train_dataset = datasets.ImageFolder(root=opt.data_folder,
											transform=TwoCropTransform(train_transform))
	else:
		raise ValueError(opt.dataset)

	train_sampler = None
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
		num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
	print(f'len dataset {opt.dataset}: {len(train_dataset)}')
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=opt.batch_size, shuffle=False,
		num_workers=opt.num_workers, pin_memory=True)

	return train_loader, val_loader


def set_model(opt):
	model = SupConResNet(name=opt.model, feat_dim=512, use_webface_pretrain=opt.use_webface_pretrain)  # todo modify classes for pretrained
	device_id = opt.device_id
	criterion = SupConLoss(temperature=opt.temp, device_id=device_id)

	# enable synchronized Batch Normalization
	if opt.syncBN:
		model = apex.parallel.convert_syncbn_model(model)

	if torch.cuda.is_available():
		# if torch.cuda.device_count() > 1:
		#     model.encoder = torch.nn.DataParallel(model.encoder, opt.device_ids)
		model = model.cuda(device_id)
		criterion = criterion.cuda(device_id)
		cudnn.benchmark = True

	return model, criterion

@torch.no_grad()
def validate(val_loader, model, criterion, opt):
	model.eval()
	batch_time = AverageMeter()
	losses = AverageMeter()
	for idx, (images, labels) in enumerate(val_loader):
		images = torch.cat([images[0], images[1]], dim=0)
		if torch.cuda.is_available():
			device_id = opt.device_id
			images = images.cuda(device=device_id, non_blocking=True)
			labels = labels.cuda(device=device_id, non_blocking=True)
		bsz = labels.shape[0]
		# compute loss
		features = model(images)
		f1, f2 = torch.split(features, [bsz, bsz], dim=0)
		features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
		if opt.method == 'SupCon':
			loss = criterion(features, labels)
		elif opt.method == 'SimCLR':
			loss = criterion(features)
		else:
			raise ValueError('contrastive method not supported: {}'.
							 format(opt.method))

		# update metric
		losses.update(loss.item(), bsz)

	return losses.avg


def train(train_loader, model, criterion, optimizer, scheduler, epoch, opt):
	"""one epoch training"""
	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	end = time.time()
	for idx, (images, labels) in enumerate(train_loader):
		data_time.update(time.time() - end)

		images = torch.cat([images[0], images[1]], dim=0)
		if torch.cuda.is_available():
			device_id = opt.device_id
			images = images.cuda(device=device_id, non_blocking=True)
			labels = labels.cuda(device=device_id, non_blocking=True)
		bsz = labels.shape[0]

		# warm-up learning rate
		# warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

		# compute loss
		features = model(images)
		f1, f2 = torch.split(features, [bsz, bsz], dim=0)
		features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
		if opt.method == 'SupCon':
			loss = criterion(features, labels)
		elif opt.method == 'SimCLR':
			loss = criterion(features)
		else:
			raise ValueError('contrastive method not supported: {}'.
							 format(opt.method))

		# update metric
		losses.update(loss.item(), bsz)

		optimizer.zero_grad()
		loss.backward()
		# grads = [torch.max(p.grad) for p in model.parameters() if p.grad is not None]
		# print(f'******** \nmax grad: {torch.max(torch.tensor(grads))}\n ********')
		optimizer.step()
		scheduler.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# print info
		if (idx + 1) % opt.print_freq == 0:
			print('Train: [{0}][{1}/{2}]\t'
				  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
				   epoch, idx + 1, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))
			sys.stdout.flush()

	return losses.avg


def main():
	opt = parse_option()
	trial_name = f"m{opt.model}_lr{opt.learning_rate}_dflr-{opt.dflr}_decay{opt.weight_decay}_bs{opt.batch_size}_ep{opt.epochs}_opt-AdamW_sch-linwp{opt.warm_epochs}_trial{opt.trial}"
	writer = SummaryWriter(osp.join('runs/supcon',trial_name))
	log.info(f"***********\n TRIAL: {trial_name}\n STARTS!***********")
	print(f"***********\n TRIAL: {trial_name}\n STARTS!***********")

	# build data loader
	train_loader, val_loader = set_loader(opt)

	# build model and criterion
	model, criterion = set_model(opt)

	# build optimizer
	optimizer = set_optimizer(opt, model)
	total_steps = len(train_loader) * opt.epochs
	scheduler_getter = transformers.get_cosine_schedule_with_warmup if opt.cosine else transformers.get_linear_schedule_with_warmup
	scheduler = scheduler_getter(optimizer=optimizer, num_warmup_steps=opt.warm_epochs*len(train_loader), num_training_steps=total_steps)

	# tensorboard
	# logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

	# training routine
	for epoch in range(1, opt.epochs + 1):
		# adjust_learning_rate(opt, optimizer, epoch)

		# train for one epoch
		time1 = time.time()
		loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, opt)
		time2 = time.time()
		print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
		val_loss = validate(val_loader, model, criterion, opt)
		print(f'epoch {epoch}| train loss: {loss} | val loss: {val_loss}')
		writer.add_scalar("Loss/train", loss, epoch)
		writer.add_scalar("Loss/val", val_loss, epoch)

		# tensorboard logger
		# logger.log_value('loss', loss, epoch)
		# logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

		if epoch % opt.save_freq == 0:
			writer.flush()
			save_file = os.path.join(
				opt.save_folder, 'webface_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))  # inceptionresnetv1 pretrained using casia-webface
			save_model(model, optimizer, opt, epoch, save_file)       

	writer.close()
	# save the last model
	save_file = os.path.join(
		opt.save_folder, 'last.pth')
	save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
	main()
