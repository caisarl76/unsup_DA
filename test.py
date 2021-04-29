import argparse
from collections import OrderedDict
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model.factory import get_model, load_model
from dataset.factory import get_dataset
from dataset.get_dataset import get_dataset
from utils.train_utils import test

root = '/media/hd/jihun/dsbn_result/'

data_pth_dict = {'officehome': 'OfficeHomeDataset_10072016', 'domainnet': 'domainnet'}
domain_dict = {'officehome': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}



def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='directory where dataset exists',
                        default='domainnet', type=str)
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/try1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--trg-domain', help='target training dataset', default='clipart')
    parser.add_argument('--src-domain', help='target training dataset', default='sketch')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=100, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'domainnet':
        num_domain = 6
        num_classes = 345
    elif args.dataset == 'officehome':
        num_domain = 4
        num_classes = 65

    _, trg_sup_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root, domain=args.trg_domain,
                                             ssl=False)

    trg_num = domain_dict[args.dataset][args.trg_domain]

    model = load_model(args.model_name, in_features=num_classes, num_classes=num_classes,
                       num_domains=num_domain, pretrained=True)
    model.load_state_dict(torch.load(args.model_path)['model'])

    model = model.cuda(args.gpu)
    _, acc = test(args, model, trg_sup_val, domain_dict[args.dataset][args.trg_domain])
    print(acc)

if __name__ == '__main__':
    main()
