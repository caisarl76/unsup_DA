# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
from os.path import join
import argparse
from time import time

# import tensorflow  # needs to call tensorflow before torch, otherwise crush

sys.path.append('Utils')
from Utils.logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('Dataset')
from network.network import Network
# from jigsaw.network.network import Network

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy
from Dataset.data_loader import DataLoader

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('--data-path', type=str, help='Path to Imagenet folder',
                    default='/data/jihun/OfficeHomeDataset_10072016/')
parser.add_argument('--domain', type=str, help='domain of dataset', default='RealWorld', )
parser.add_argument('--model-path', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=30, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=40, type=int, help='batch size')
parser.add_argument('--save-dir', default='result', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

# from ImageDataLoader import DataLoader

save_root = '/media/hd/jihun/dsbn_result/results/dsbn_ori/jigsaw_result/'

def main():
    save_dir = join(save_root, args.save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.gpu is not None:
        print(('Using GPU %d' % args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('CPU mode')

    print('Process number: %d' % (os.getpid()))

    ## DataLoader initialize ILSVRC2012_train_processed
    trainpath = join(args.data_path, args.domain)
    train_data = DataLoader(trainpath, split='train', classes=args.classes, ssl=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.cores)

    valpath = join(args.data_path, args.domain)
    val_data = DataLoader(valpath, split='validation', classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                             num_workers=args.cores)

    iter_per_epoch = train_data.N / args.batch
    print('Images: train %d, validation %d' % (train_data.N, val_data.N))

    # Network initialize
    net = Network(args.classes)
    if args.gpu is not None:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger = Logger(join(save_root,args.save_dir ,'train'))
    logger_test = Logger(join(save_root,args.save_dir ,'test'))

    ############## TESTING ###############
    if args.evaluate:
        test(net, criterion, None, val_loader, 0)
        return

    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d' % (args.lr, args.batch, args.classes)))
    print(('Checkpoint: ' + args.save_dir))

    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start
    best_acc = -1
    for epoch in range(args.epochs):
        if epoch % 10 == 0 and epoch > 0:
            net, acc = test(net, criterion, logger_test, val_loader, steps)
            if(best_acc < acc):
                net.save(join(save_dir, 'best_model.pth'))
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time() - end)
            if len(batch_time) > 100:
                del batch_time[0]

            images = Variable(images)
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time() - t)
            if len(net_time) > 100:
                del net_time[0]

            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            # acc = prec1[0]
            acc = prec1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            steps += 1

            if steps % 1000 == 0:
                filename = join(save_dir, ('%06d.pth.tar'%(steps)))
                net.save(filename)
                print('Saved: ' + args.save_dir)

            end = time()


    ###########################################################################################################
    #                                   classifier finetune                                                   #
    ###########################################################################################################
    finetune_model = Network(65)
    pretrained_dict = {k:v for k, v in net.state_dict().items() if k in finetune_model.state_dict()}
    finetune_model.state_dict().update(pretrained_dict)
    # finetune_model.classifier.

def test(net, criterion, logger, val_loader, steps):
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())
    acc = np.mean(accuracy)
    if logger is not None:
        logger.scalar_summary('accuracy', acc, steps)
    print('TESTING: %d), Accuracy %.2f%%' % (steps, acc))
    net.train()
    return net, acc

if __name__ == "__main__":
    main()
