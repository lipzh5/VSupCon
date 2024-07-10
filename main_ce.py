from __future__ import print_function

import os
import os.path as osp
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import transformers
from torch.utils.tensorboard import SummaryWriter

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model, eval_meld
from networks.resnet_big import SupCEResNet

try:
	import apex
	from apex import amp, optimizers
except ImportError:
	pass


def parse_option():
	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--print_freq', type=int, default=100,
						help='print frequency')
	parser.add_argument('--save_freq', type=int, default=50,
						help='save frequency')
	parser.add_argument('--batch_size', type=int, default=256,
						help='batch_size')
	parser.add_argument('--num_workers', type=int, default=16,
						help='num of workers to use')
	parser.add_argument('--epochs', type=int, default=500,
						help='number of training epochs')

	# optimization
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help='learning rate')
	parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
						help='where to decay lr, can be a list')
	parser.add_argument('--lr_decay_rate', type=float, default=0.1,
						help='decay rate for learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4,
						help='weight decay')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='momentum')

	# model dataset
	parser.add_argument('--model', type=str, default='inceptionresnetv1') # 'resnet50'
	parser.add_argument('--dataset', type=str, default= 'affwild2', # 'cifar10',
						choices=['cifar10', 'cifar100', 'affwild2'], help='dataset')

	# other setting
	parser.add_argument('--cosine', action='store_true',
						help='using cosine annealing')
	parser.add_argument('--syncBN', action='store_true',
						help='using synchronized batch normalization')
	parser.add_argument('--warm', action='store_true',
						help='warm-up for large batch training')
	parser.add_argument('--warm_epochs', type=int, default=1)
	parser.add_argument('--trial', type=str, default='0',
						help='id for recording multiple runs')
	parser.add_argument('--device_id', type=int, default=0)
	parser.add_argument('--image_size', type=int, default=160)  # 160 for affwild2, 32 for cifar
	parser.add_argument('--data_dsr', type=int, default=1, help='data set downsampling ratio, e.g., 1 stands for original, 2 stands for original//2')
	parser.add_argument('--use_webface_pretrain', action='store_true')
	parser.add_argument('--weight_init', action='store_true', help='using customized weight init')
	opt = parser.parse_args()

	# set the path according to the environment
	opt.data_folder = './datasets/'
	opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
	opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

	iterations = opt.lr_decay_epochs.split(',')
	opt.lr_decay_epochs = list([])
	for it in iterations:
		opt.lr_decay_epochs.append(int(it))

	opt.model_name = 'SupCE_{}_{}_ep_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
		format(opt.dataset, opt.model, opt.epochs, opt.learning_rate, opt.weight_decay,
			   opt.batch_size, opt.trial)

	if opt.cosine:
		opt.model_name = '{}_cosine'.format(opt.model_name)

	# warm-up for large-batch training,
	if opt.batch_size > 256:
		opt.warm = True
	if opt.warm:
		opt.model_name = '{}_warm'.format(opt.model_name)
		opt.warmup_from = 0.01
		opt.warm_epochs = 1
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

	if opt.dataset == 'cifar10':
		opt.n_cls = 10
	elif opt.dataset == 'cifar100':
		opt.n_cls = 100
	elif opt.dataset == 'affwild2':
		opt.n_cls = 7
	else:
		raise ValueError('dataset not supported: {}'.format(opt.dataset))

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
			mean = (0.4654, 0.3532, 0.3217)
			std = (0.2334, 0.2003, 0.1902)
		else:  # original size 
			mean = (0.4652, 0.3531, 0.3215)
			std = (0.2348, 0.2019, 0.1918)
	else:
		raise ValueError('dataset not supported: {}'.format(opt.dataset))
	normalize = transforms.Normalize(mean=mean, std=std)

	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(size=opt.image_size, scale=(0.2, 1.)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])

	val_transform = transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])

	if opt.dataset == 'cifar10':
		train_dataset = datasets.CIFAR10(root=opt.data_folder,
										 transform=train_transform,
										 download=True)
		val_dataset = datasets.CIFAR10(root=opt.data_folder,
									   train=False,
									   transform=val_transform)
	elif opt.dataset == 'cifar100':
		train_dataset = datasets.CIFAR100(root=opt.data_folder,
										  transform=train_transform,
										  download=True)
		val_dataset = datasets.CIFAR100(root=opt.data_folder,
										train=False,
										transform=val_transform)
	elif opt.dataset == 'affwild2':
		train_dataset = get_affwild2_dataset('train', train_transform, opt.data_dsr)
		opt.size=160
		val_dataset = get_affwild2_dataset('val', val_transform, opt.data_dsr)
	else:
		raise ValueError(opt.dataset)

	train_sampler = None
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
		num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=opt.batch_size, shuffle=False,
		num_workers=8, pin_memory=True)

	return train_loader, val_loader


def set_model(opt):
	model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, use_webface_pretrain=opt.use_webface_pretrain, weight_init=opt.weight_init)
	criterion = torch.nn.CrossEntropyLoss()

	# enable synchronized Batch Normalization
	if opt.syncBN:
		model = apex.parallel.convert_syncbn_model(model)

	if torch.cuda.is_available():
		# if torch.cuda.device_count() > 1:
		# 	model = torch.nn.DataParallel(model)
		model = model.cuda(opt.device_id)
		criterion = criterion.cuda(opt.device_id)
		cudnn.benchmark = True

	return model, criterion


def train(train_loader, model, criterion, optimizer, scheduler, epoch, opt):
	"""one epoch training"""
	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	end = time.time()
	for idx, (images, labels) in enumerate(train_loader):
		data_time.update(time.time() - end)

		images = images.cuda(device=opt.device_id, non_blocking=True)
		labels = labels.cuda(device=opt.device_id, non_blocking=True)
		bsz = labels.shape[0]

		# warm-up learning rate
		# warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

		# compute loss
		output = model(images)
		loss = criterion(output, labels)

		# update metric
		losses.update(loss.item(), bsz)
		acc1, acc5 = accuracy(output, labels, topk=(1, 5))
		top1.update(acc1[0], bsz)

		# SGD
		optimizer.zero_grad()
		loss.backward()
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
				  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
				  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				   epoch, idx + 1, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1))
			sys.stdout.flush()

	return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
	"""validation"""
	model.eval()

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	all_logits, all_truths = [], []
	with torch.no_grad():
		end = time.time()
		for idx, (images, labels) in enumerate(val_loader):
			images = images.float().cuda(opt.device_id)
			labels = labels.cuda(opt.device_id)
			bsz = labels.shape[0]

			# forward
			output = model(images)
			loss = criterion(output, labels)
			all_logits.append(output)
			all_truths.append(labels)

			# update metric
			losses.update(loss.item(), bsz)
			acc1, acc5 = accuracy(output, labels, topk=(1, 5))
			top1.update(acc1[0], bsz)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if idx % opt.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					   idx, len(val_loader), batch_time=batch_time,
					   loss=losses, top1=top1))

	print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
	wf = eval_meld(torch.cat(all_logits), torch.cat(all_truths))
	print(f'WF : {wf} \n *******')

	return losses.avg, top1.avg, wf


def main():
	best_acc = 0
	opt = parse_option()
	trial_name = f"m{opt.model}_winit{opt.weight_init}_lr{opt.learning_rate}_decay{opt.weight_decay}_bs{opt.batch_size}_ep{opt.epochs}_webface{int(opt.use_webface_pretrain)}_trial{opt.trial}"
	writer = SummaryWriter(osp.join('runs',trial_name))
	# log.info(f"***********\n TRIAL: {trial_name}\n STARTS!***********")
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
		adjust_learning_rate(opt, optimizer, epoch)

		# train for one epoch
		time1 = time.time()
		loss, train_acc = train(train_loader, model, criterion, optimizer, scheduler, epoch, opt)
		time2 = time.time()
		print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

		# tensorboard logger
		# logger.log_value('train_loss', loss, epoch)
		# logger.log_value('train_acc', train_acc, epoch)
		# logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
		writer.add_scalar("Loss/train", loss, epoch)
		

		# evaluation
		loss, val_acc, wf = validate(val_loader, model, criterion, opt)
		writer.add_scalar("Loss/val", loss, epoch)
		writer.add_scalar("WF/val", wf, epoch)
		# logger.log_value('val_loss', loss, epoch)
		# logger.log_value('val_acc', val_acc, epoch)

		if val_acc > best_acc:
			best_acc = val_acc

		if epoch % opt.save_freq == 0:
			writer.flush()
			save_file = os.path.join(
				opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
			save_model(model, optimizer, opt, epoch, save_file)

	# save the last model
	save_file = os.path.join(
		opt.save_folder, 'last.pth')
	save_model(model, optimizer, opt, opt.epochs, save_file)

	print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
	main()
