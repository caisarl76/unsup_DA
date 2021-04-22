from model.resnetdsbn import resnet50dsbn
import torch
from torch import nn as nn
from torch.nn import init
from collections import OrderedDict


def main():
    pth = '/home/vision/jhkim/best_model.ckpt'
    src_bn = 'bns.' + (str)(0)
    trg_bn = 'bns.' + (str)(1)

    model = resnet50dsbn(pretrained=False, in_features=65, num_classes=65, num_domains=4)
    model.load_state_dict(torch.load(pth)['model'])
    weight_dict = OrderedDict()

    for name, p in model.named_parameters():
        if name == 'layer4.0.bn1.bns.1.bias':
            print(p)

    for name, p in model.named_parameters():
        if (src_bn in name):
            # print(name)
            new_name = name.replace(src_bn, trg_bn)
            # print(new_name)

            weight_dict[new_name] = p

    model.load_state_dict(weight_dict, strict=False)
    for name, p in model.named_parameters():
        if name == 'layer4.0.bn1.bns.1.weight':
            print(p)


if __name__ == '__main__':
    main()
