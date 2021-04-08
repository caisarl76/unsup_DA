import os, sys, numpy as np
from os.path import join
import argparse
from time import time
from collections import OrderedDict
from sklearn.metrics import confusion_matrix

sys.path.append('Utils')
from Utils.logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('Dataset')
from network.AlexnetDSBN import AlexnetDSBN as Network

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy
from Dataset.data_loader import DataLoader

# model1 = torch.load('/home/vision/jhkim/results/dsbn_ori/jigsaw_result/jigsaw_ssl_r_c/stage1/best_model.pth')
# model2 = torch.load('/home/vision/jhkim/results/dsbn_ori/jigsaw_result/jigsaw_ssl_r_c/stage2/best_model.pth')

if __name__ == '__main__':

    model = Network(65)
    # for param in model.parameters():
    #     print(param.requires_grad)

    for name, p in model.named_parameters():
        if ('fc' in name) or ('bns.1' in name):
            # if('weight' in name):

            p.requires_grad = True
            print(name)
        else:
            p.requires_grad = False
            print('-------', name)



    # print('------------------------------------------------------------')
    #
    # for param in model.parameters():
    #     print(param.requires_grad)
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    #
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    # for param in model.parameters():
    #     print(param.requires_grad)
    #     print(type(param))
    #
    # for param in model.parameters():
    #     print(param.requires_grad)

    # for name in model1:
    #     if (name in model2):
    #         # print('----', name)
    #         # print()
    #         if ('weight' in name):
    #             print('----', name)
    #             print(model1[name].sum())
    #             # break
    #             if(model1[name].sum() != model1[name].sum()):
    #                 print('----------------------',name)
    #     else:
    #         # print('-')
    #         continue
