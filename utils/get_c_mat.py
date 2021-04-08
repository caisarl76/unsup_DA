import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import MNIST, SVHN
from dataset.factory import get_dataset
from model.lenet import DSBNLeNet
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

svhn_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(),
    transforms.ToTensor()
])
save_root = '/home/vision/jhkim/results/dsbn_ori/digits-result/'


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', help='directory to save models', default='result/mnist_svhn/try2', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/mnist_svhn/try2',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='lenetdsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='mnist')
    parser.add_argument('--trg-domain', help='target training dataset', default='svhn')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=128, type=int)
    parser.add_argument("--iters", type=int, default=[10000, 10000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=3e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args



def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    stage = args.stage

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

    svhn_train = SVHN(root='/data/jihun/SVHN', transform=svhn_transform, download=True)
    svhn_val = SVHN(root='/data/jihun/SVHN', split='test', transform=svhn_transform, download=True)
    mnist_train = MNIST('/data/jihun/MNIST', train=True, transform=mnist_transform, download=True)
    mnist_val = MNIST('/data/jihun/MNIST', train=False, transform=mnist_transform, download=True)

    dataloader1 = util_data.DataLoader(svhn_val, batch_size=args.batch_size, shuffle=True, num_workers=5,
                                                drop_last=True, pin_memory=True)
    dataloader1_iter = enumerate(dataloader1)