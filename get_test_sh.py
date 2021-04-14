import argparse
import os
from collections import OrderedDict
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms

domain_dict = {'r': 'RealWorld', 'a': 'Art', 'c': 'Clipart', 'p': 'Product'}

base = 'CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path '

dirs = ['rot_sup/resnet50',
        # 'byol_finetune',
        'rot_ssl/byol',
        'rot_ssl/resnet50',
        'rot_ssl_byol/resnet50',
        'rot_sup/resnet50'
        ]


def main():
    f = open('run_test.sh', 'w')
    f.write('#!/bin/bash \n')

    for dir in dirs:
        path = join('/result', dir)
        exp_list = os.listdir(path)
        for exp in exp_list:
            print(exp)
            src, trg = exp.split('_')
            src = domain_dict[src]
            trg = domain_dict[trg]
            print(dir, exp, src, trg)

            line = base + join(path, exp, 'stage2/best_model.ckpt') + ' --domain ' + src + '\n'
            print(line)
            f.write('echo exp: %s \n' % (join(path, exp)))
            f.write('echo src: %s \n' % (src))
            f.write(line)

            line = base + join(path, exp, 'stage2/best_model.ckpt') + ' --domain ' + trg + '\n'

            f.write('echo trg: %s \n' % (trg))
            f.write(line)
            # break

    f.close()


if __name__ == '__main__':
    main()
