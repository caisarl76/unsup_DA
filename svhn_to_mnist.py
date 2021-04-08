import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import MNIST, SVHN
from dataset.factory import get_dataset
from model.lenet import DSBNLeNet
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

svhn_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(),
    transforms.ToTensor()
])
save_root = '/media/hd/jihun/dsbn_result/results/dsbn_ori/digits-result/'


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', help='directory to save models', default='result/mnist_svhn/try2', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/mnist_svhn/try2',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='lenetdsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='mnist')
    parser.add_argument('--trg-domain', help='target training dataset', default='svhn')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=128, type=int)
    parser.add_argument("--iters", type=int, default=[10000, 10000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=3e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    stage = args.stage

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

    svhn_train = SVHN(root='/data/jihun/SVHN', transform=svhn_transform, download=True)
    svhn_val = SVHN(root='/data/jihun/SVHN', split='test', transform=svhn_transform, download=True)
    mnist_train = MNIST('/data/jihun/MNIST', train=True, transform=mnist_transform, download=True)
    mnist_val = MNIST('/data/jihun/MNIST', train=False, transform=mnist_transform, download=True)

    if (stage == 1):
        save_dir = join(save_root, args.save_dir, 'stage1')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if (args.trg_domain == 'mnist'):
            train_dataset = mnist_train
            val_dataset = mnist_val
        else:
            train_dataset = svhn_train
            val_dataset = svhn_val

        train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5,
                                                drop_last=True, pin_memory=True)

        train_dataloader_iters = enumerate(train_dataloader)
        model = DSBNLeNet(num_classes=10, in_features=0, num_domains=2)
        model.train(True)
        model = model.cuda(args.gpu)
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
        ce_loss = nn.CrossEntropyLoss()
        domain_num = 0

        best_accuracy = 0.0
        best_accuracies_each_c = []
        best_mean_val_accuracies = []
        best_total_val_accuracies = []

        writer = SummaryWriter()
        for i in range(args.iters[0]):
            try:
                _, (x_s, y_s) = train_dataloader_iters.__next__()
            except StopIteration:
                train_dataloader_iters = enumerate(train_dataloader)
                _, (x_s, y_s) = train_dataloader_iters.__next__()

            optimizer.zero_grad()
            # lr_scheduler(optimizer, i)

            x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
            # x_s = x_s.cuda(args.gpu)
            domain_idx = torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu)
            pred, f = model(x_s, domain_num * domain_idx, with_ft=True)
            loss = ce_loss(pred, y_s)
            # print(loss)
            writer.add_scalar("Train Loss", loss, i)
            loss.backward()
            optimizer.step()

            if (i % 500 == 0 and i != 0):
                # print('------%d val start' % (i))
                model.eval()
                total_val_accuracies = []
                mean_val_accuracies = []
                val_accuracies_each_c = []
                model.eval()

                val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers, drop_last=True, pin_memory=True)
                val_dataloader_iter = enumerate(val_dataloader)

                pred_vals = []
                y_vals = []
                x_val = None
                y_val = None

                with torch.no_grad():
                    for j, (x_val, y_val) in val_dataloader_iter:
                        y_vals.append(y_val.cpu())
                        x_val = x_val.cuda(args.gpu)
                        y_val = y_val.cuda(args.gpu)

                        pred_val = model(x_val, domain_num * torch.ones_like(y_val), with_ft=False)

                        pred_vals.append(pred_val.cpu())

                pred_vals = torch.cat(pred_vals, 0)
                y_vals = torch.cat(y_vals, 0)
                total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
                val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                               class_idx=c, topk=(1,))[0]))
                                       for c, c_name in enumerate(val_dataset.classes)]

                mean_val_accuracy = float(
                    torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))
                total_val_accuracies.append(total_val_accuracy)
                val_accuracies_each_c.append(val_accuracy_each_c)
                mean_val_accuracies.append(mean_val_accuracy)

                val_accuracy = float(torch.mean(torch.FloatTensor(total_val_accuracies)))
                print('%d th iteration accuracy: %f ' % (i, val_accuracy))
                del x_val, y_val, pred_val, pred_vals, y_vals
                del val_dataloader_iter

                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

                model.train(True)  # train mode
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_accuracies_each_c = val_accuracies_each_c
                    best_mean_val_accuracies = mean_val_accuracies
                    best_total_val_accuracies = total_val_accuracies
                    # print('%d iter val acc %.3f' % (i, val_accuracy))
                    model_dict = {'model': model.cpu().state_dict()}
                    optimizer_dict = {'optimizer': optimizer.state_dict()}

                    # save best checkpoint
                    io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=True)

                model = model.cuda(args.gpu)
        if args.proceed:
            stage += 1
    if (stage == 2):
        save_dir = join(save_root, args.save_dir, 'stage2')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if (args.src_domain == 'mnist'):
            train_dataset = mnist_train
            val_dataset = mnist_val
        else:
            train_dataset = svhn_train
            val_dataset = svhn_val

        train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
        train_dataloader_iters = enumerate(train_dataloader)

        model = DSBNLeNet(num_classes=10, in_features=0, num_domains=2)
        if (args.proceed):
            model.load_state_dict(torch.load(join(save_root, args.save_dir, 'stage1', 'best_model.ckpt'))['model'])
        else:
            model.load_state_dict(torch.load(save_root, args.model_path)['model'])

        for name, p in model.named_parameters():
            if ('fc' in name) or 'bns.1' in name:
                p.requires_grad = True
                continue
            else:
                p.requires_grad = False

        torch.nn.init.xavier_uniform_(model.fc1.weight)
        torch.nn.init.xavier_uniform_(model.fc2.weight)

        model.train(True)
        model = model.cuda(args.gpu)

        params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                      double_bias_lr=True, base_weight_factor=0.1)

        optimizer = optim.Adam(params, betas=(0.9, 0.999))
        ce_loss = nn.CrossEntropyLoss()

        writer = SummaryWriter()
        domain_num = stage - 1
        print('domain_num, stage: ', domain_num, stage)

        best_accuracy = 0.0
        best_accuracies_each_c = []
        best_mean_val_accuracies = []
        best_total_val_accuracies = []

        for i in range(args.iters[stage - 1]):
            try:
                _, (x_s, y_s) = train_dataloader_iters.__next__()
            except StopIteration:
                train_dataloader_iters = enumerate(train_dataloader)
                _, (x_s, y_s) = train_dataloader_iters.__next__()
            optimizer.zero_grad()
            # lr_scheduler(optimizer, i)

            x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
            domain_idx = torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu)
            pred, f = model(x_s, domain_num * domain_idx, with_ft=True)
            loss = ce_loss(pred, y_s)
            writer.add_scalar("Train Loss", loss, i)
            loss.backward()
            optimizer.step()

            if (i % 500 == 0 and i != 0):
                # print('------%d val start' % (i))
                model.eval()
                total_val_accuracies = []
                mean_val_accuracies = []
                val_accuracies_each_c = []
                model.eval()

                val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers, drop_last=True, pin_memory=True)
                val_dataloader_iter = enumerate(val_dataloader)

                pred_vals = []
                y_vals = []
                x_val = None
                y_val = None
                # print('------------------------dataload------------------------')
                with torch.no_grad():
                    for j, (x_val, y_val) in val_dataloader_iter:
                        y_vals.append(y_val.cpu())
                        x_val = x_val.cuda(args.gpu)
                        y_val = y_val.cuda(args.gpu)

                        pred_val = model(x_val, domain_num * torch.ones_like(y_val), with_ft=False)

                        pred_vals.append(pred_val.cpu())

                pred_vals = torch.cat(pred_vals, 0)
                y_vals = torch.cat(y_vals, 0)
                total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
                val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                               class_idx=c, topk=(1,))[0]))
                                       for c, c_name in enumerate(val_dataset.classes)]

                mean_val_accuracy = float(
                    torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))
                total_val_accuracies.append(total_val_accuracy)
                val_accuracies_each_c.append(val_accuracy_each_c)
                mean_val_accuracies.append(mean_val_accuracy)

                val_accuracy = float(torch.mean(torch.FloatTensor(total_val_accuracies)))
                print('%d th iteration accuracy: %f ' % (i, val_accuracy))
                del x_val, y_val, pred_val, pred_vals, y_vals
                del val_dataloader_iter

                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

                model.train(True)  # train mode
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_accuracies_each_c = val_accuracies_each_c
                    best_mean_val_accuracies = mean_val_accuracies
                    best_total_val_accuracies = total_val_accuracies
                    # print('%d iter val acc %.3f' % (i, val_accuracy))
                    model_dict = {'model': model.cpu().state_dict()}
                    optimizer_dict = {'optimizer': optimizer.state_dict()}

                    # save best checkpoint
                    io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=True)

                model = model.cuda(args.gpu)
        if (args.proceed):
            stage += 1

    if (stage == 3):

        if (args.trg_domain == 'mnist'):
            val_dataset = mnist_val
        else:
            val_dataset = svhn_val

        val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5,
                                              drop_last=True, pin_memory=True)

        val_dataloader_iter = enumerate(val_dataloader)

        model = DSBNLeNet(num_classes=10, in_features=0, num_domains=2)
        if (args.proceed):
            model.load_state_dict(torch.load(join(save_root, args.save_dir, 'stage2', 'best_model.ckpt'))['model'])
        else:
            model.load_state_dict(torch.load(save_root, args.model_path)['model'])
        model = model.cuda(args.gpu)

        pred_vals = []
        y_vals = []
        domain_num = 0

        with torch.no_grad():
            for j, (x_val, y_val) in val_dataloader_iter:
                y_vals.append(y_val.cpu())
                x_val = x_val.cuda(args.gpu)
                y_val = y_val.cuda(args.gpu)

                pred_val = model(x_val, domain_num * torch.ones_like(y_val), with_ft=False)

                pred_vals.append(pred_val.cpu())

        pred_vals = torch.cat(pred_vals, 0)
        y_vals = torch.cat(y_vals, 0)
        total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
        val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                       class_idx=c, topk=(1,))[0]))
                               for c, c_name in enumerate(val_dataset.classes)]

        print(total_val_accuracy)
        print(val_accuracy_each_c)


if __name__ == '__main__':
    main()
