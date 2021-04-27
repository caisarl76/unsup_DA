import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import OFFICEHOME_multi
from rot_dataset.rot_dataset import rot_dataset
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from utils.train_utils import get_optimizer_params, normal_train, test
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils
from collections import OrderedDict

domain_dict = {'RealWorld': 1, 'Clipart': 0}
root = '/media/hd/jihun/dsbn_result/'


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/OfficeHomeDataset_10072016', type=str)
    parser.add_argument('--save-root', help='directory to save models', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='ssl_result/0315_3/', type=str)

    parser.add_argument('--model-path', help='directory to save models',
                        default='ssl_result/0315_3/stage1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='RealWorld')
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart', nargs='+')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[20000, 20000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    save_root = root
    if (args.save_root):
        save_root = args.save_root

    ### 1. train encoder with rotation task ###
    save_dir = join(save_root, args.save_dir, 'stage1')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model = get_rot_model(args.model_name, num_domains=4)

    train_dataset = rot_dataset(args.data_root, 1, [args.trg_domain], 'train')
    val_dataset = rot_dataset(args.data_root, 1, [args.trg_domain], 'val')
    model = normal_train(args, model, train_dataset, val_dataset, args.iters[0], save_dir, args.trg_domain)

    ### 2. train classifier with classification task ###
    pre = torch.load(join(save_dir, 'best_model.ckpt'))


    model.load_state_dict(pre, strict=False)

    del pre

    model = get_model(args.model_name, 65, 65, 4)
    for name, p in model.named_parameters():
        p.requires_grad = False

    model.fc1.weight.requires_grad = True
    model.fc2.weight.requires_grad = True
    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)

    train_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='train')
    val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='val')

    save_dir = join(save_root, args.save_dir, 'stage2')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model = normal_train(args, model, train_dataset, val_dataset, args.iters[1], save_dir, args.trg_domain)



if __name__ == '__main__':
    main()
