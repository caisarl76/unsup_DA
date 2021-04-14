import argparse
import os
from collections import OrderedDict
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms

domain_dict = {'r': 'RealWorld', 'a': 'Art', 'c': 'Clipart', 'p': 'Product'}

base = 'CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-name resnet50dsbn --model-path '

dirs = ['rot_sup/resnet50',
        # 'byol_finetune',
        'rot_ssl/byol',
        'rot_ssl/resnet50',
        'rot_ssl_byol/resnet50',
        'rot_sup/resnet50'
        ]


def main():
    for dir in dirs:
        path = join('/result', dir)
        exp_list = os.listdir(path)
        for exp in exp_list:
            print(exp)
            src, trg = exp.split('_')
            src = domain_dict[src]
            trg = domain_dict[trg]
            print(dir, exp, src, trg)

            line = base + join(path, exp, 'stage2/best_model.ckpt') +'--domain ' + src
            print(line)

            # break
if __name__ == '__main__':
    main()
