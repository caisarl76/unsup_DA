from model.resnetdsbn import resnet50dsbn, resnet18dsbn
from model.factory import get_model
import torch
from torch import nn as nn
from torch.nn import init
from collections import OrderedDict


def main():
    pth = '/results/officehome/stage1/rot/Art/best_model.ckpt'
    src_bn = 'bns.' + (str)(0)
    trg_bn = 'bns.' + (str)(1)

    model = get_model('resnet50dsbn', in_features=256, num_classes=4, num_domains=4,
                       pretrained=True)
    pre = torch.load(pth)['model']
    model.load_state_dict(pre)
    print(model)

    model.load_state_dict(pre, strict=False)
    print(len(model.parameters()))
    print(len(model.named_parameters()))
    src_bn = 'bns.' + (str)(0)
    trg_bn = 'bns.' + (str)(1)

    weight_dict = OrderedDict()
    for name, p in model.named_parameters():
        if (trg_bn in name):
            weight_dict[name] = p
            new_name = name.replace(trg_bn, src_bn)
            weight_dict[new_name] = p
        elif (src_bn in name):
            continue
        else:
            weight_dict[name] = p
    model.load_state_dict(weight_dict, strict=False)

    for name, p in model.named_parameters():
        p.requires_grad = False

    model.fc1.weight.requires_grad = True
    model.fc2.weight.requires_grad = True
    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)

if __name__ == '__main__':
    main()

    # model = load_model(model_name='resnet18dsbn', num_classes=345, in_features=345, num_domains=6, pretrained=False, cut_conv=2)
    # for name, p in model.named_parameters():
    #     print(name)
