from model.resnetdsbn import resnet50dsbn, resnet18dsbn
from model.factory import get_model
import torch
from torch import nn as nn
from torch.nn import init
from collections import OrderedDict


def main():
    pth = '/media/hd/jihun/dsbn_result/new/rot_sup/p_c/stage2/best_model.ckpt'
    src_bn = 'bns.' + (str)(0)
    trg_bn = 'bns.' + (str)(1)

    model = get_model('resnet50dsbn', in_features=65, num_classes=65, num_domains=4,
                       pretrained=True)
    pre = torch.load(pth)['model']
    # model.load_state_dict(pre)
    # print(model)
    model.eval()
    before_forward = {
        'running_mean': model.bn1.bns[0].running_mean.clone(),
        'running_var': model.bn1.bns[0].running_var.clone(),
        'weight': model.bn1.bns[0].weight.clone(),
        'bias': model.bn1.bns[0].bias.clone(),
    }
    model(torch.randn([3, 3, 224, 224]), torch.zeros(3, dtype=torch.long))

    print(torch.all(before_forward['running_mean'] == model.bn1.bns[0].running_mean))
    print(torch.all(before_forward['running_var'] == model.bn1.bns[0].running_var))
    print(torch.all(before_forward['weight'] == model.bn1.bns[0].weight))
    print(torch.all(before_forward['bias'] == model.bn1.bns[0].bias))

    for name, p in model.named_parameters():
        print(name)
    return



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
