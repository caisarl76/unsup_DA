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

save_root = '/media/hd/jihun/dsbn_result/new/'
domain_dict = {'RealWorld': 1, 'Clipart': 0, 'Product': 0, 'Art': 0}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description='Train jigsaw puzzle archi with sup task')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--trg-domain', type=str, help='target domain of dataset', default='Clipart')
    parser.add_argument('--src-domain', type=str, help='source domain of dataset', default='RealWorld')
    parser.add_argument('--model-path',
                        default='../../results/dsbn_ori/jigsaw_result/class30_RealWorld/best_model.pth',
                        type=str, help='Path to pretrained model')
    parser.add_argument('--classes', default=[30, 65], type=int, help='Number of permutation to use', nargs='+')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs for training')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
    parser.add_argument('--batch', default=40, type=int, help='batch size')
    parser.add_argument('--save-dir', default='dsbn_jigsaw', type=str, help='save_dir')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')
    parser.add_argument('--cores', default=3, type=int, help='number of CPU core for loading')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set, No training')

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)
    args = parser.parse_args()
    return args


def test(args, model, logger, val_loader, steps, domain_num, save_dir):
    print('Evaluating network.......')
    accuracy = []
    model.eval()
    y_s = []
    pred_s = []

    for i, (x, y) in enumerate(val_loader):
        x = Variable(x)
        if args.gpu is not None:
            x = x.cuda()

        # Forward + Backward + Optimize
        outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
        outputs = outputs.cpu().data
        if (steps == -1):
            pred_s += outputs.argmax(axis=1)
            y_s += y

        prec1, _ = compute_accuracy(outputs, y, topk=(1, 5))
        accuracy.append(prec1.item())
    acc = np.mean(accuracy)

    if (steps == -1):
        c_mat = confusion_matrix(y_true=y_s, y_pred=pred_s)
        np.save(join(save_dir, 'stage2_c_mat.npy'), c_mat)
        print('confusion matrix saved at: ', join(save_dir, 'stage2_c_mat.npy'))
    if logger is not None:
        logger.scalar_summary('accuracy', acc, steps)
    print('TESTING: %d), Accuracy %.2f%%' % (steps, acc))
    model.train(True)
    return model, acc


def ssl_train(args, model, train_data, val_data, save_dir, domain_num):
    logger_test = Logger(save_dir + '/test')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.cores)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                             num_workers=args.cores)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4)

    iter_per_epoch = train_data.N / args.batch
    print('Images: train %d, validation %d' % (train_data.N, val_data.N))
    print(('Start training: lr %f, batch size %d, classes %d' % (args.lr, args.batch, args.classes[0])))
    batch_time, net_time = [], []
    steps = args.iter_start
    best_acc = -1
    for epoch in range(int(args.iter_start / iter_per_epoch), args.epochs):
        if epoch % 10 == 0 and epoch > 0:
            print('Evaluating network.......')
            accuracy = []
            model.eval()
            for i, (x, y, _) in enumerate(val_loader):
                x = Variable(x)
                if args.gpu is not None:
                    x = x.cuda()

                # Forward + Backward + Optimize
                outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
                outputs = outputs.cpu().data

                prec1, prec5 = compute_accuracy(outputs, y, topk=(1, 5))
                accuracy.append(prec1.item())
            acc = np.mean(accuracy)
            if (best_acc < acc):
                best_acc = acc
                filename = join(save_dir, 'best_model.pth')
                model.save(filename)

            if logger_test is not None:
                logger_test.scalar_summary('accuracy', acc, steps)
            print('TESTING: %d), Accuracy %.2f%%' % (steps, acc))
            model.train()

        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)

        end = time()
        for i, (x, y, original) in enumerate(train_loader):
            batch_time.append(time() - end)
            if len(batch_time) > 100:
                del batch_time[0]

            images = Variable(x)
            labels = Variable(y)
            if args.gpu is not None:
                x = x.cuda()
                y = y.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
            net_time.append(time() - t)
            if len(net_time) > 100:
                del net_time[0]

            prec1, prec5 = compute_accuracy(outputs.cpu().data, y.cpu().data, topk=(1, 5))
            # acc = prec1[0]
            acc = prec1
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())
            steps += 1

            if steps % 1000 == 0:
                filename = 'step_%06d.pth.tar' % (steps)
                filename = join(save_dir, filename)
                model.save(filename)
                print('model saved at: ', filename)

            end = time()


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
            model, acc = test(args, model, logger_test, val_loader, steps, domain_num, save_dir)
            if (best_acc < acc):
                best_acc = acc
                filename = join(save_dir, 'best_model.pth')
                model.save(filename)

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
                outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
                net_time.append(time() - t)
                if len(net_time) > 100:
                    del net_time[0]

                prec1 = compute_accuracy(outputs.cpu().data, y.cpu().data, topk=(1,))
                acc = prec1[0]
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                steps += 1

                if (best_acc < acc):
                    best_acc = acc
                    filename = join(save_dir, 'best_model.pth')
                    model.save(filename)

                if steps % 1000 == 0:
                    filename = 'step_%06d.pth.tar' % (steps)
                    filename = join(save_dir, filename)
                    model.save(filename)
                    print('steps: %d' % (steps))

                end = time()

    steps = -1
    model, acc = test(args, model, logger_test, val_loader, steps, domain_num, save_dir)
    print('final acc: %0.3f' % (acc))
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

    stage = args.stage
    if (stage == 1):
        model = Network(args.classes[stage - 1])
        model.train(True)
        if args.gpu is not None:
            model.cuda()

        save_dir = join(save_root, args.save_dir, 'stage1')
        os.makedirs(save_dir, exist_ok=True)
        print('save dir: ', save_dir)

        trainpath = join(args.data_path, args.trg_domain)
        train_data = DataLoader(trainpath, split='train', classes=args.classes[stage - 1], ssl=True)
        valpath = join(args.data_path, args.trg_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes[stage - 1], ssl=True)
        domain_num = domain_dict[args.trg_domain]
        ssl_train(args=args, model=model, train_data=train_data, val_data=val_data, save_dir=save_dir,
                  domain_num=domain_num)

        if (args.proceed):
            stage += 1

    if (stage == 2):
        save_dir = join(save_root, args.save_dir, 'stage2')
        os.makedirs(save_dir, exist_ok=True)
        print('save dir: ', save_dir)

        model = Network(args.classes[stage - 1])

        if (args.proceed):
            model_path = join(save_dir.replace('stage2', 'stage1'), 'best_model.pth')
        else:
            model_path = args.model_path

        pre = torch.load(model_path)
        new_pre = OrderedDict()
        for name in pre:
            if ("classifier" in name):
                continue
            else:
                new_pre[name] = pre[name]

        model.load_state_dict(new_pre, strict=False)

        for name, p in model.named_parameters():
            if ('bns.1' in name) or('fc' in name):
                p.requires_grad = True
                print(name)
                continue
            else:
                p.requires_grad = False
        torch.nn.init.xavier_uniform_(model.fc6.fc6_s1.weight)
        torch.nn.init.xavier_uniform_(model.fc7.fc7.weight)
        torch.nn.init.xavier_uniform_(model.classifier.fc8.weight)
        model.train(True)

        if args.gpu is not None:
            model.cuda()

        trainpath = join(args.data_path, args.src_domain)
        train_data = DataLoader(trainpath, split='train', classes=args.classes[stage - 1], ssl=False)
        valpath = join(args.data_path, args.src_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes[stage - 1], ssl=False)

        train(args=args, model=model, train_data=train_data, val_data=val_data, save_dir=save_dir,
              domain_num=domain_dict[args.src_domain])

        if (args.proceed):
            stage += 1

    if stage == 3:
        model = Network(args.classes[stage - 2])
        if args.proceed:
            model_path = join(save_root, args.save_dir, 'stage2', 'best_model.pth')
        else:
            model_path = args.model_path
        model.load_state_dict(torch.load(model_path))
        if args.gpu is not None:
            model.cuda()
        model.eval()

        logger = Logger(join(save_dir, 'test'))

        valpath = join(args.data_path, args.trg_domain)
        val_data = DataLoader(valpath, split='train', classes=args.classes[stage - 2], ssl=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=True,
                                                 num_workers=args.cores)
        y_s = []
        pred_s = []
        accuracy = []
        for i, (x, y) in enumerate(val_loader):
            x = Variable(x)
            if args.gpu is not None:
                x = x.cuda()

            # Forward + Backward + Optimize
            outputs = model(x, domain_num * torch.ones(x.shape[0], dtype=torch.long).cuda())
            outputs = outputs.cpu().data
            y_s += y
            pred_s += outputs.argmax(axis=1)

            prec1, _ = compute_accuracy(outputs, y, topk=(1, 5))
            accuracy.append(prec1.item())

        c_mat = confusion_matrix(y_true=y_s, y_pred=pred_s)
        np.save(join(save_dir, 'stage3_c_mat.npy'), c_mat)
        print('confusion matrix saved at: ', join(save_dir, 'stage3_c_mat.npy'))

        print('TESTING: ), Accuracy %.2f%%' % (np.mean(accuracy)))
        # model.train()


if __name__ == '__main__':
    main()
