import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data as util_data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pickle
import os
from os.path import join as join
from collections import OrderedDict

import sys
sys.path.append('../')
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from dataset.datasets import OFFICEHOME_multi
from utils.train_utils import get_optimizer_params
from utils import io_utils, eval_utils

save_root = '/media/hd/jihun/dsbn_result/new/'
weight_path = 'byol_r50_bs256_accmulate16_ep300-5df46722.pth'
domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--save-root', help='directory to save models', default=None, type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='byol_finetune/r', type=str)
    parser.add_argument('--freeze', help='freeze encoder', action='store_true')
    parser.add_argument('--domain', help='training dataset', default='RealWorld')

    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iter", type=int, default=550, help="iterations.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    args = parser.parse_args(args=args, namespace=namespace)
    return args

def load_byol_weight(model, byol_path):
    bn_list = ['bns.0', 'bns.1', 'bns.2', 'bns.3']
    byol_weight = torch.load(byol_path)['state_dict']
    new_dict = OrderedDict()

    for name in byol_weight:
        if ('bn' in name):
            for bn in bn_list:
                new_split = name.split('.')
                new_split.insert(-1, bn)
                new_name = '.'.join(new_split)
                new_dict[new_name] = byol_weight[name]
        elif ('downsample.1' in name):
            for bn in bn_list:
                new_split = name.split('.')
                new_split.insert(-1, bn)
                new_name = '.'.join(new_split)
                new_dict[new_name] = byol_weight[name]
        #         print(name)
        else:
            new_dict[name] = byol_weight[name]

    model.load_state_dict(new_dict, strict=False)
    return model

def train(args, model, train_dataset, val_dataset, save_dir, domain_num):
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    train_dataloader_iters = enumerate(train_dataloader)

    model.train(True)
    model = model.cuda(args.gpu)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=True, base_weight_factor=0.1)

    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    ce_loss = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=join(save_dir, 'logs'))
    print('domain_num: ', domain_num)
    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(args.iter):
        try:
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        except StopIteration:
            train_dataloader_iters = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iters.__next__()

        optimizer.zero_grad()

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

            # train mode
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
            model.train(True)
            model = model.cuda(args.gpu)

        if (i % 10000 == 0 and i != 0):
            print('%d iter complete' % (i))
    writer.flush()
    writer.close()

    return




def main():
    args = parse_args()
    if(args.save_root):
        save_root = args.save_root
        print('save root: ', save_root)
    save_dir = join(save_root, args.save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    weight_path = 'byol_r50_bs256_accmulate16_ep300-5df46722.pth'
    model = get_model('resnet50dsbn', 65, 65, 4, pretrained=True)
    model = load_byol_weight(model, weight_path)
    domain_num = domain_dict[args.domain]

    if(args.freeze):
        bn_name = 'bns.' + (str)(domain_num)
        for name, p in model.named_parameters():
            if ('fc' in name) or bn_name in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    train_dataset = OFFICEHOME_multi(args.data_root, 1, [args.domain], split='train')
    val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.domain], split='val')

    train(args, model, train_dataset, val_dataset, save_dir, domain_num)

if __name__ == '__main__':
    main()
