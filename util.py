from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(f'correct: {correct.shape}, {correct[:k].shape}')
            # print(f'viewed tensor: {correct[:k].view(-1).shape}')
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_paramsgroup(opt, model, warmup=False):
	no_decay = ['bias', 'LayerNorm.weight']
	pre_train_lr = opt.learning_rate
	encoder_params = list(map(id, model.encoder.parameters()))
	params = []
	warmup_params = []
	for name, param in model.named_parameters():
		lr = pre_train_lr * 10
		weight_decay = opt.weight_decay
		if id(param) in encoder_params:
			lr = pre_train_lr
		if any(nd in name for nd in no_decay):
			weight_decay = 0
		params.append({
			'params': param,
			'lr': lr,
			'weight_decay': weight_decay
		})
		warmup_params.append({
			'params':
			param,
			'lr':
			pre_train_lr / 4 if id(param) in encoder_params else lr,
			'weight_decay':
			weight_decay
		})
	if warmup:
		return warmup_params
	params = sorted(params, key=lambda x: x['lr'])
	return params

def set_optimizer(opt, model):
    # return optim.AdamW(get_paramsgroup(opt, model)) if opt.dflr else optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
  
    # optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer




def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def eval_meld(results, truths, test=False, return_all=False):
    test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
    test_truth = truths.cpu().detach().numpy()  #（num_utterance）
    predicted_label = []
    true_label = []
    for i in range(test_preds.shape[0]):
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
        true_label.append(test_truth[i])
    wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
    # if test:
    f1_each_label = f1_score(true_label, predicted_label, average=None)
    test_str = 'TEST' if test else 'EVAL'
    print(f'**{test_str}** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
    if return_all:
        return wg_av_f1, f1_each_label
    return wg_av_f1 


