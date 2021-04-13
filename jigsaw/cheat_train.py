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
from network.AlexnetDSBN import AlexnetDSBN
from Dataset.data_loader import DataLoader

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy

# save_root = '/media/hd/jihun/dsbn_result/results/dsbn_ori/jigsaw_result/'
save_root = '/media/hd/jihun/dsbn_result/new/'

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

domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}
transform = {'train': train_transform, 'val': val_transform, 'test': val_transform}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description='Train jigsaw puzzle archi with sup task')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--trg_domain', type=str, help='target domain of dataset', default='Clipart')
    parser.add_argument('--src_domain', type=str, help='source domain of dataset', default='RealWorld')
    # parser.add_argument('--model-path',
    #                     default='../../results/dsbn_ori/jigsaw_result/class30_RealWorld/jps_035_001000.pth.tar',
    #                     type=str,
    #                     help='Path to pretrained model')
    parser.add_argument('--classes', default=65, type=int, help='Number of permutation to use')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--epochs', default=140, type=int, help='number of total epochs for training')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
    parser.add_argument('--batch', default=40, type=int, help='batch size')
    parser.add_argument('--save-dir', default='cheat_stage1', type=str, help='save_dir')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')
    parser.add_argument('--cores', default=4, type=int, help='number of CPU core for loading')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set, No training')

    parser.add_argument('--all-fc', help='freeze all fc layers on stage 2', action='store_true')
    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)
    args = parser.parse_args()
    return args


def test(args, model, logger, val_loader, steps, domain_num):
    print('Evaluating network.......')
    accuracy = []
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        x = Variable(x)
        if args.gpu is not None:
            x = x.cuda()

        # Forward + Backward + Optimize
        outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
        outputs = outputs.cpu().data

        prec1, _ = compute_accuracy(outputs, y, topk=(1, 5))
        accuracy.append(prec1.item())
    acc = np.mean(accuracy)
    if logger is not None:
        logger.scalar_summary('accuracy', acc, steps)
    print('TESTING: %d), Accuracy %.2f%%' % (steps, acc))
    model.train(True)
    return model, acc


def load_pretrained_weights(args):
    model = AlexnetDSBN(args.classes)
    pre = torch.load(args.model_path)
    new_pre = OrderedDict()
    for p in pre:
        if ('classifier' in p):
            # print('----', p)
            continue
        else:
            new_pre[p] = pre[p]
    model.load_state_dict(new_pre, strict=False)

    for name, p in model.named_parameters():
        if ('classifier' in name):
            continue
        else:
            p.requires_grad = False

    torch.nn.init.xavier_uniform_(model.classifier.fc8.weight)
    del new_pre

    return model


def train(args, model, train_data, val_data, save_dir, domain_num):
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.cores)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                             num_workers=args.cores)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4)
    logger = Logger(join(save_dir, 'train'))
    logger_test = Logger(join(save_dir, 'test'))

    batch_time, net_time = [], []
    steps = args.iter_start
    best_acc = -1
    for epoch in range(args.epochs):
        if epoch % 10 == 0 and epoch > 0:
            model, acc = test(args, model, logger_test, val_loader, steps, domain_num)
            if (best_acc < acc):
                best_acc = acc
                filename = join(save_dir, 'best_model.pth')
                model.save(filename)
                print('model saved')
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        for i, (x, y) in enumerate(train_loader):
            # print(i, images.shape, labels)
            batch_time.append(time() - end)
            if len(batch_time) > 100:
                del batch_time[0]

            x = Variable(x)
            y = Variable(y)
            if args.gpu is not None:
                x = x.cuda()
                y = y.cuda()
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                t = time()
                outputs = model(x, 0 * torch.ones(x.shape[0], dtype=torch.long).cuda())
                net_time.append(time() - t)
                if len(net_time) > 100:
                    del net_time[0]

                prec1 = compute_accuracy(outputs.cpu().data, y.cpu().data, topk=(1,))
                acc = prec1[0]
                # acc = prec1

                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                loss = float(loss.cpu().data.numpy())
                steps += 1

                if steps % 1000 == 0:
                    filename = 'jps_%03i_%06d.pth.tar' % (epoch, steps)
                    filename = join(save_dir, filename)
                    model.save(filename)
                    print('Saved: ' + save_dir)

                end = time()
    return


def main():
    args = parse_args()
    if args.gpu is not None:
        print(('Using GPU %d' % args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('CPU mode')

    print('Process number: %d' % (os.getpid()))

    # model = load_pretrained_weights(args)

    stage = args.stage
    if (stage == 1):
        model = AlexnetDSBN(args.classes)
        model.train(True)
        if args.gpu is not None:
            model.cuda()

        save_dir = join(save_root, args.save_dir, 'stage1')
        print('save dir: ', save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        trainpath = join(args.data_path, args.trg_domain)
        train_data = DataLoader(trainpath, split='train', classes=args.classes, ssl=False)
        valpath = join(args.data_path, args.trg_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes, ssl=False)
        domain_num = domain_dict[args.trg_domain]
        train(args, model, train_data, val_data, save_dir, domain_num)
        if (args.proceed):
            stage += 1

    if (stage == 2):
        model = AlexnetDSBN(args.classes)

        save_dir = join(save_root, args.save_dir, 'stage2')
        print('save dir: ', save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        trainpath = join(args.data_path, args.src_domain)
        train_data = DataLoader(trainpath, split='train', classes=args.classes, ssl=False)
        valpath = join(args.data_path, args.src_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes, ssl=False)
        domain_num = domain_dict[args.src_domain]

        if (args.proceed):
            model_path = join(save_dir.replace('stage2', 'stage1'), 'best_model.pth')
        print(model_path)
        model.load_state_dict(torch.load(model_path))
        # if (args.all_fc):
        for name, p in model.state_dict().items():
            if ('fc' in name) or ('bns.1' in name):
                continue
            else:
                p.requires_grad = False
        torch.nn.init.xavier_uniform_(model.fc6.fc6_s1.weight)
        torch.nn.init.xavier_uniform_(model.fc7.fc7.weight)
        torch.nn.init.xavier_uniform_(model.classifier.fc8.weight)
        # else:
        #     for name, p in model.named_parameters():
        #         if ('classifier' in name) or ('bns.1' in name):
        #             continue
        #         else:
        #             p.requires_grad = False
        #     torch.nn.init.xavier_uniform_(model.classifier.fc8.weight)
        model.train(True)
        if args.gpu is not None:
            model.cuda()

        train(args, model, train_data, val_data, save_dir, domain_num)
        if (args.proceed):
            stage += 1

    if (stage == 3):
        model = AlexnetDSBN(args.classes)
        if (args.proceed):
            model_path = join(save_root, args.save_dir, 'stage2', 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        # model.train(True)
        if args.gpu is not None:
            model.cuda()

        valpath = join(args.data_path, args.trg_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes, ssl=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                                 num_workers=args.cores)
        domain_num = domain_dict[args.trg_domain]
        accuracy = []
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            x = Variable(x)
            if args.gpu is not None:
                x = x.cuda()

            # Forward + Backward + Optimize
            outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
            outputs = outputs.cpu().data

            prec1, _ = compute_accuracy(outputs, y, topk=(1, 5))
            accuracy.append(prec1.item())

        print('TESTING: ), Accuracy %.2f%%' % (np.mean(accuracy)))
        # model.train()


if __name__ == '__main__':
    main()
