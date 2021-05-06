import argparse
import os
from os.path import join as join
import torch
from collections import OrderedDict
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from utils.train_utils import normal_train, test

from dataset.get_dataset import get_dataset

# domain_dict = {'RealWorld': 1, 'Clipart': 0}
root = '/media/hd/jihun/dsbn_result/'

data_pth_dict = {'officehome': 'OfficeHomeDataset_10072016', 'domainnet': 'domainnet'}
domain_dict = {'officehome': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='directory where dataset exists',
                        default='officehome', type=str)
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/', type=str)
    parser.add_argument('--save-root', help='directory to save models', default='/results/result/rot_ssl/', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='domainnet/clipart_sketch', type=str)
    parser.add_argument('--model-name', default='resnet50dsbn', type=str)
    parser.add_argument('--model-path', default='path for stage2 train', type=str)
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart')
    parser.add_argument('--src-domain', help='target training dataset', default='RealWorld')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[30005, 10005], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--lr-scheduler', '-lrsche', dest='lr_scheduler',
                        help='learning_rate scheduler [Lambda/Multiplicate/Step/Multistep/Expo', type=str)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument("--ssl", action='store_true')
    parser.add_argument("--only1", action='store_true')
    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    save_root = root
    stage = args.stage

    num_domain = 4
    num_classes = 65

    if (args.save_root):
        save_root = args.save_root

    trg_ssl_train, trg_ssl_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root,
                                             domain=args.trg_domain,
                                             ssl=True)
    src_ssl_train, src_ssl_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root,
                                             domain=args.src_domain,
                                             ssl=True)

    model = get_model(args.model_name, in_features=256, num_classes=4, num_domains=num_domain,
                      pretrained=True)










if __name__ == '__main__':
    main()




