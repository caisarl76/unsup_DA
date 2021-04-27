import math
from os.path import join as join
import torch
from torch import nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import defaultdict
from torch.autograd import Variable
from torch.utils import data as util_data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from utils import io_utils, eval_utils

domain_lib = {'office-home': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}



def test(args, model, val_dataset, domain_num):
    val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_dataloader_iter = enumerate(val_dataloader)

    val_accs_each_c = []
    pred_ys = []
    y_vals = []
    x_val = None
    y_val = None

    model.eval()
    with torch.no_grad():
        for j, (x_val, y_val) in val_dataloader_iter:
            y_vals.append(y_val.cpu())
            x_val = x_val.cuda(args.gpu)
            y_val = y_val.cuda(args.gpu)

            pred_y = model(x_val, domain_num * torch.ones_like(y_val), with_ft=False)
            pred_ys.append(pred_y.cpu())

    pred_ys = torch.cat(pred_ys, 0)
    y_vals = torch.cat(y_vals, 0)
    val_acc = float(eval_utils.accuracy(pred_ys, y_vals, topk=(1,))[0])
    val_acc_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_ys, y_vals,
                                                              class_idx=c, topk=(1,))[0]))
                      for c, c_name in enumerate(val_dataset.classes)]
    return model, val_acc


def normal_train(args, model, train_dataset, val_dataset, iter, save_dir, domain, freeze=False):
    domain_dict = domain_lib[args.dataset]
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    train_dataloader_iters = enumerate(train_dataloader)

    model.train(True)
    model = model.cuda(args.gpu)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=True, base_weight_factor=0.1)
    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    ce_loss = nn.CrossEntropyLoss()
    lr_scheduler = LRScheduler(args.learning_rate, 5e-6, 5000,
                               num_steps=iter,
                               alpha=10, beta=0.75, double_bias_lr=True,
                               base_weight_factor=0.1)

    if (args.lr_scheduler):
        if (args.lr_scheduler == 'Step'):
            optimizer = optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.999))
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    writer = SummaryWriter(log_dir=join(save_dir, 'logs'))
    domain_num = domain_dict[domain]
    print('domain: %s, %d' % (domain, domain_num))
    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(iter):
        try:
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        except StopIteration:
            train_dataloader_iters = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iters.__next__()

        optimizer.zero_grad()

        x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
        domain_idx = torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu)
        pred, f = model(x_s, domain_num * domain_idx, with_ft=True)
        loss = ce_loss(pred, y_s)
        writer.add_scalar("Train Loss", loss, i)
        if not freeze:
            loss.backward()
            optimizer.step()

            if (args.lr_scheduler):
                lr_scheduler.step()
            else:
                lr_scheduler(optimizer, i)


        if (i % 500 == 0 and i != 0):
            model, acc = test(args, model, val_dataset, domain_num)
            # print('%d iter || val acc: %0.3f' % (i, acc))
            writer.add_scalar("Val Accuracy", acc, i)
            if acc > best_accuracy:
                best_accuracy = acc
                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=True)
            if (i % 1000 == 0 and i != 0):
                print('%d iter accuracy: %0.3f' % (i, acc))

            model.train(True)
            model = model.cuda(args.gpu)

    model, acc = test(args, model, val_dataset, domain_num)
    print('final acc: %0.3f' % (acc))

    writer.flush()
    writer.close()

    return model


def adaptation_factor(p, gamma=10):
    p = max(min(p, 1.0), 0.0)
    den = 1.0 + math.exp(-gamma * p)
    lamb = 2.0 / den - 1.0
    return min(lamb, 1.0)


def lr_poly(base_lr, i_iter, alpha=10, beta=0.75, num_steps=250000):
    if i_iter < 0:
        return base_lr
    return base_lr / ((1 + alpha * float(i_iter) / num_steps) ** (beta))


def semantic_loss_calc(x, y, mean=True):
    loss = (x - y) ** 2
    if mean:
        return torch.mean(loss)
    else:
        return loss


def l2_loss_calc(x):
    return torch.sum(x ** 2) / 2


def wce_loss(x, y, instance_weights=None):
    if instance_weights is not None:
        return torch.mean(instance_weights * F.cross_entropy(x, y, reduce=False))
    else:
        return F.cross_entropy(x, y)


def wbce_loss(x, y, instance_weights=None):
    if instance_weights is not None:
        return torch.mean(instance_weights * F.binary_cross_entropy_with_logits(x, y, reduce=False))
    else:
        return F.binary_cross_entropy_with_logits(x, y)


def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d):
            # init.xavier_normal_(m.weight)
            m.weight.data.normal_(0, 0.01).clamp_(min=-0.02, max=0.02)
            try:
                m.bias.data.zero_()
            except AttributeError:
                # no bias
                pass
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01).clamp_(min=-0.02, max=0.02)
            try:
                m.bias.data.zero_()
            except AttributeError:
                # no bias
                pass
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, 0, 0.01)


def noise_injection_to_label(labels, num_classes, noise_prob=0.1):
    sample_prob = torch.zeros(labels.shape[0], num_classes).fill_(noise_prob / (num_classes - 1))
    sample_prob[torch.arange(labels.shape[0]), labels] = 1 - noise_prob
    sampler = torch.distributions.categorical.Categorical(sample_prob)
    return sampler.sample()


def get_optimizer_params(modules, lr, weight_decay=0.0005, double_bias_lr=True, base_weight_factor=0.1):
    weights = []
    biases = []
    base_weights = []
    base_biases = []
    if isinstance(modules, list):
        for module in modules:
            for key, value in dict(module.named_parameters()).items():
                if value.requires_grad:
                    if 'fc' in key or 'score' in key:
                        if 'bias' in key:
                            biases += [value]
                        else:
                            weights += [value]
                    else:
                        if 'bias' in key:
                            base_biases += [value]
                        else:
                            base_weights += [value]
    else:
        module = modules
        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if 'fc' in key or 'score' in key:
                    if 'bias' in key:
                        biases += [value]
                    else:
                        weights += [value]
                else:
                    if 'bias' in key:
                        base_biases += [value]
                    else:
                        base_weights += [value]
    if base_weight_factor:
        params = [
            {'params': weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': biases, 'lr': lr * (1 + float(double_bias_lr))},
            {'params': base_weights, 'lr': lr * base_weight_factor, 'weight_decay': weight_decay},
            {'params': base_biases, 'lr': lr * base_weight_factor * (1 + float(double_bias_lr))},
        ]
    else:
        params = [
            {'params': base_weights + weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': base_biases + biases, 'lr': lr * (1 + float(double_bias_lr))},
        ]
    return params


def KD_loss_with_label_calc(outputs, labels, teacher_outputs, alpha=1.0, temperature=1.0):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities!
    """
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def KL_loss_calc(outputs, teacher_outputs, temperature=1.0):
    """
    Compute the KL divergence (KL) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = (T * T) * nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                       F.softmax(teacher_outputs / T, dim=1))

    return KD_loss


def KL_u_p_loss(outputs):
    # KL(u||p)
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduce=False).sum(dim=1)
    return instance_losses


def L2_u_p_loss(outputs):
    # ||u-p||^2
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = Variable(uniform_tensors / num_classes).cuda()
    instance_losses = semantic_loss_calc(uniform_dists, outputs, mean=False)
    return instance_losses


class LRScheduler:
    def __init__(self, learning_rate, warmup_learning_rate=0.0, warmup_steps=2000, num_steps=200000, alpha=10,
                 beta=0.75,
                 double_bias_lr=False, base_weight_factor=False):
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.double_bias_lr = double_bias_lr
        self.base_weight_factor = base_weight_factor

    def __call__(self, optimizer, i_iter):
        if i_iter < self.warmup_steps:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.learning_rate

        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 4:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
            optimizer.param_groups[2]['lr'] = self.base_weight_factor * lr_poly(lr, lr_i_iter, alpha=self.alpha,
                                                                                beta=self.beta,
                                                                                num_steps=self.num_steps)
            optimizer.param_groups[3]['lr'] = (1 + float(self.double_bias_lr)) * self.base_weight_factor * lr_poly(lr,
                                                                                                                   lr_i_iter,
                                                                                                                   alpha=self.alpha,
                                                                                                                   beta=self.beta,
                                                                                                                   num_steps=self.num_steps)
        else:
            raise RuntimeError('Wrong optimizer param groups')

    def current_lr(self, i_iter):
        if i_iter < self.warmup_steps:
            return self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            return lr_poly(self.learning_rate, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)


class Monitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self._cummulated_losses = defaultdict(lambda: 0.0)
        self._total_counts = defaultdict(lambda: 0)

    def update(self, losses_dict):
        for key in losses_dict:
            self._cummulated_losses[key] += losses_dict[key]
            self._total_counts[key] += 1

    @property
    def cummulated_losses(self):
        return self._cummulated_losses

    @property
    def total_counts(self):
        return self._total_counts

    @property
    def losses(self):
        losses = {}
        for k, v in self._cummulated_losses.items():
            if self._total_counts[k] > 0:
                losses[k] = v / float(self._total_counts[k])
            else:
                losses[k] = 0.0
        return losses

    def __repr__(self):
        sorted_loss_keys = sorted([k for k in self._cummulated_losses.keys()])
        losses = self.losses
        repr_str = ''
        for key in sorted_loss_keys:
            repr_str += ', {0}={1:.4f}'.format(key, losses[key])
        return repr_str[2:]


def one_hot_encoding(y, n_classes):
    tensor_size = [y.size(i) for i in range(len(y.size()))]
    if tensor_size[-1] != 1:
        tensor_size += [1]
    tensor_size = tuple(tensor_size)
    y_one_hot = torch.zeros(tensor_size[:-1] + (n_classes,)).to(y.device).scatter_(len(tensor_size) - 1,
                                                                                   y.view(tensor_size), 1)
    return y_one_hot
