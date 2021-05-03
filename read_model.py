from model.resnetdsbn import resnet50dsbn, resnet18dsbn
from model.factory import load_model
import torch
from torch import nn as nn
from torch.nn import init
from collections import OrderedDict


def main():
    pth = '/result/officehome/stage1/rot/RealWorld/best_model.ckpt'
    src_bn = 'bns.' + (str)(0)
    trg_bn = 'bns.' + (str)(1)

    model = load_model('resnet50dsbn', in_features=256, num_classes=4, num_domains=4,
                       pretrained=True)
    pre = torch.load(pth)['model']
    model.load_state_dict(pre)
    print(model)
    # print(model)
    # new_pre = OrderedDict()
    # for key in pre.keys():
    #     if 'fc' in key:
    #         print(key)
    #     else:
    #         new_pre[key] = pre[key]
    #
    # model.load_state_dict(new_pre, strict=False)
    # for name, p in model.named_parameters():
    #     print(name)

    weight_dict = OrderedDict()

    # for name, p in model.named_parameters():
    #     if name == 'layer4.0.bn1.bns.1.bias':
    #         print(p)
    #
    # for name, p in model.named_parameters():
    #     if (src_bn in name):
    #         print(name)
    #         new_name = name.replace(src_bn, trg_bn)
    #         print(new_name)
    #
    #         weight_dict[new_name] = p
    #
    # model.load_state_dict(weight_dict, strict=False)



if __name__ == '__main__':
    main()

    # model = load_model(model_name='resnet18dsbn', num_classes=345, in_features=345, num_domains=6, pretrained=False, cut_conv=2)
    # for name, p in model.named_parameters():
    #     print(name)
