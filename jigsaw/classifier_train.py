import os, sys, numpy as np
# sys.path.append('../dataset')
from os.path import join
import argparse
from time import time
from collections import OrderedDict
from torchvision import transforms

# import tensorflow  # needs to call tensorflow before torch, otherwise crush

sys.path.append('Utils')
from Utils.logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('Dataset')
from network.network import Network
from Dataset.data_loader import DataLoader

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

domain_dict = {'RealWorld': 1, 'Clipart': 0}
transform = {'train': train_transform, 'val': val_transform, 'test': val_transform}

save_root = '/media/hd/jihun/dsbn_result/new/'

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('--data-path', type=str, help='Path to Imagenet folder',
                    default='/data/jihun/OfficeHomeDataset_10072016/')
parser.add_argument('--domain', type=str, help='domain of dataset', default='RealWorld')
parser.add_argument('--model-path',
                    default='../../results/dsbn_ori/jigsaw_result/class30_RealWorld/jps_035_001000.pth.tar', type=str,
                    help='Path to pretrained model')
parser.add_argument('--classes', default=65, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=40, type=int, help='batch size')
parser.add_argument('--save-dir', default='result', type=str, help='save_dir')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()


def test(model, criterion, logger, val_loader, steps):
    print('Evaluating network.......')
    accuracy = []
    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = model(images)
        outputs = outputs.cpu().data

        prec1, _ = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())

    if logger is not None:
        logger.scalar_summary('accuracy', np.mean(accuracy), steps)
    print('TESTING: %d), Accuracy %.2f%%' % (steps, np.mean(accuracy)))
    model.train()
    return model


def load_pretrained_weights(args):
    model = Network(args.classes)
    model_path = args.model_path.replace('RealWorld', args.domain)
    pre = torch.load(args.model_path)
    new_pre = OrderedDict()
    for p in pre:
        if ('classifier' in p):
            # print('----', p)
            continue
        else:
            new_pre[p] = pre[p]
    model.load_state_dict(new_pre, strict=False)

    for name, p in model.state_dict().items():
        if ('classifier' in name):
            continue
        else:
            p.requires_grad = False

    torch.nn.init.xavier_uniform_(model.classifier.fc8.weight)
    del new_pre

    return model


def main():
    if args.gpu is not None:
        print(('Using GPU %d' % args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('CPU mode')

    print('Process number: %d' % (os.getpid()))
    model = load_pretrained_weights(args)
    model.train(True)
    if args.gpu is not None:
        model.cuda()
    trainpath = join(args.data_path, args.domain)
    train_data = DataLoader(trainpath, split='train', classes=args.classes, ssl=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.cores)
    valpath = join(args.data_path, args.domain)
    val_data = DataLoader(valpath, split='train', classes=args.classes, ssl=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                             num_workers=args.cores)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    logger = Logger(join(save_root, args.save_dir, 'train'))
    logger_test = Logger(join(save_root, args.save_dir, 'test'))

    batch_time, net_time = [], []
    steps = args.iter_start
    for epoch in range(args.epochs):
        if epoch % 10 == 0 and epoch > 0:
            model = test(model, criterion, logger_test, val_loader, steps)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        for i, (images, labels) in enumerate(train_loader):
            # print(i, images.shape, labels)
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
                outputs = model(images)
                net_time.append(time() - t)
                if len(net_time) > 100:
                    del net_time[0]

                prec1 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1,))
                acc = prec1[0]
                # acc = prec1

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss = float(loss.cpu().data.numpy())

                if steps % 20 == 0:
                    print(
                        (
                                '[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' % (
                            epoch + 1, args.epochs, steps,
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss, acc)))
                steps += 1

                if steps % 1000 == 0:
                    filename = 'jps_%03i_%06d.pth.tar' % (epoch, steps)
                    filename = join(save_root, args.save_dir, filename)
                    model.save(filename)
                    print('Saved: ' + args.save_dir)

                end = time()


if __name__ == "__main__":
    main()
