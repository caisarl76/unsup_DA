import argparse
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model.factory import load_model
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from utils.train_utils import get_optimizer_params, normal_train, test

from dataset.get_dataset import get_dataset

# domain_dict = {'RealWorld': 1, 'Clipart': 0}
root = '/media/hd/jihun/dsbn_result/'

data_pth_dict = {'office-home': 'OfficeHomeDataset_10072016', 'domainnet': 'domainnet'}
domain_dict = {'office-home': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}
class_dict = {'office-home': 65, 'domainnet': 345}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='directory where dataset exists',
                        default='domainnet', type=str)
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/', type=str)
    parser.add_argument('--save-root', help='directory to save models', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='ssl_result/0315_3/', type=str)

    parser.add_argument('--model-name', default='resnet50dsbn', type=str)
    parser.add_argument('--domain', help='target training dataset', default='clipart')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=100, type=int)
    parser.add_argument("--iters", type=int, default=[6, 6], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--lr-scheduler', '-lrsche', dest='lr_scheduler',
                        help='learning_rate scheduler [Lambda/Multiplicate/Step/Multistep/Expo', type=str)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    save_root = root
    if (args.save_root):
        save_root = args.save_root
    stage = args.stage

    ### 1. train encoder with rotation task ###
    save_dir = join(save_root, args.save_dir, 'stage1')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if (stage == 1):
        train_dataset, val_dataset = get_dataset(dataset=args.dataset, dataset_root=args.data_root, domain=args.domain,
                                                 ssl=True)

        model = load_model(args.model_name, in_features=256, num_classes=4, num_domains=6, pretrained=True)
        # model = get_rot_model(args.model_name, num_domains=6)
        model = normal_train(args, model, train_dataset, val_dataset, args.iters[0], save_dir, args.domain,
                             save_model=True)

        stage += 1

    ### 2. train classifier with classification task ###
    if (stage == 2):
        train_dataset, val_dataset = get_dataset(dataset=args.dataset, dataset_root=args.data_root, domain=args.domain,
                                                 ssl=False)

        for i in range(5):
            # iter = i * 20000 + 10000
            iter = i * 2 + 1
            pre = torch.load(join(save_dir, '%d_model.ckpt'%(iter)))
            model = load_model(args.model_name, in_features=class_dict[args.dataset],
                               num_classes=class_dict[args.dataset],
                               num_domains=6, pretrained=True)

            new_pre = OrderedDict()
            for key in pre.keys():
                if 'fc' in key:
                    print(key)
                else:
                    new_pre[key] = pre[key]

            model.load_state_dict(new_pre, strict=False)

            torch.nn.init.xavier_uniform_(model.fc1.weight)
            torch.nn.init.xavier_uniform_(model.fc2.weight)
            model.fc1.weight.requires_grad = True
            model.fc2.weight.requires_grad = True

            save_dir = join(save_root, args.save_dir, 'stage2_%d'%(iter))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            model = normal_train(args, model, train_dataset, val_dataset, args.iters[1], save_dir, args.domain)

        pre = torch.load(join(save_dir, 'best_model.ckpt'))

        model = load_model(args.model_name, in_features=class_dict[args.dataset], num_classes=class_dict[args.dataset],
                           num_domains=6, pretrained=True)

        new_pre = OrderedDict()
        for key in pre.keys():
            if 'fc' in key:
                print(key)
            else:
                new_pre[key] = pre[key]

        model.load_state_dict(new_pre, strict=False)

        # for name, p in model.named_parameters():
        #     p.requires_grad = False

        torch.nn.init.xavier_uniform_(model.fc1.weight)
        torch.nn.init.xavier_uniform_(model.fc2.weight)
        model.fc1.weight.requires_grad = True
        model.fc2.weight.requires_grad = True


        save_dir = join(save_root, args.save_dir, 'stage2')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        model = normal_train(args, model, train_dataset, val_dataset, args.iters[1], save_dir, args.domain)


if __name__ == '__main__':
    main()
    # model = get_model('resnet50dsbn', 344, 344, 6)
    # print(model)
