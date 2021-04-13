# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import sys
sys.path.append('.')
sys.path.append('Utils')
sys.path.append('../Utils')
# from jigsaw.Utils.Layers import LRN
# from ..Utils.Layers import LRN

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class AlexnetDSBN(nn.Module):

    def __init__(self, classes=1000, in_features=0, num_domains=4, mmd=False):
        self.inplanes = 64
        self.in_features = in_features
        self.num_domains = num_domains
        self.num_classes = classes
        self.mmd = mmd
        super(AlexnetDSBN, self).__init__()

        self.conv1 = LRNBlock(inplanes=3, planes=96, kernel_size_1=11, kernel_size_2=3, padding=0, first=True)
        self.conv2 = LRNBlock(inplanes=96, planes=256, kernel_size_1=5, kernel_size_2=3, padding=2, groups=2,
                              first=False)
        self.conv3 = BasicBlock(inplanes=256, planes=384, first=True)
        self.conv4 = BasicBlock(inplanes=384, planes=384)
        self.conv5 = BasicBlock(inplanes=384, planes=256, last=True)

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(256 * 3 * 3, 1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1', nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(9 * 1024, 4096))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(4096, classes))

        # self.apply(weights_init)

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        # pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, domain_num):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            out = self.conv1(x[i], domain_num)
            out = self.conv2(out, domain_num)
            out = self.conv3(out, domain_num)
            out = self.conv4(out, domain_num)
            out = self.conv5(out, domain_num)
            out = self.fc6(out.view(B, -1))
            out = out.view([B, 1, -1])
            x_list.append(out)

        x = cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        if not(self.mmd):
            x = self.classifier(x)

        return x

    def printnorm(self, input, output):
        output_norm = output.data.norm()
        return output_norm


class LRNBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size_1=11, kernel_size_2=3, padding=0, groups=2, num_domains=4,
                 first=True):
        super(LRNBlock, self).__init__()
        self.first = first
        if (self.first):
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size_1, stride=2, padding=padding)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size_1, padding=padding, groups=groups)
        self.bns = DomainSpecificBatchNorm2d(planes, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_2, stride=2)
        self.lrn = LRN(local_size=5, alpha=0.0001, beta=0.75)

    def forward(self, x, domain_label):
        out = self.conv(x)
        out, _ = self.bns(out, domain_label)
        out = self.relu(out)
        out = self.pool(out)
        out = self.lrn(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num_domains=4, first=False, last=False):
        super(BasicBlock, self).__init__()
        self.first = first
        self.last = last
        self.num_domains = num_domains
        if(self.first):
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, groups=2)

        self.bn1 = DomainSpecificBatchNorm2d(planes, self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        if (self.last):
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x, domain_label):
        out = self.conv(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)
        if (self.last):
            out = self.pool(out)

        return out


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
