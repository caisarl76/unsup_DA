import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
from collections import OrderedDict
import torch
from torch.utils import data as util_data
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.rot_resnetdsbn import get_rot_model
from rot_dataset.rot_dataset import rot_dataset
from dataset.datasets import OFFICEHOME_multi
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

save_root = '/media/hd/jihun/dsbn_result/new/'

domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}

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


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--save-root', help='directory to save models', default=None, type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='result/try1', type=str)
    parser.add_argument('--ssl', help='stage 1 selfsup learning', action='store_true')
    parser.add_argument('--model-path', help='directory to save models', default='result/try1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='RealWorld')
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[30000, 10000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def train(args, model, train_dataset, val_dataset, stage, save_dir, domain_num):
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    train_dataloader_iters = enumerate(train_dataloader)

    model.train(True)
    model = model.cuda(args.gpu)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=True, base_weight_factor=0.1)

    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    ce_loss = nn.CrossEntropyLoss()

    writer = SummaryWriter()
    print('domain_num, stage: ', domain_num, stage)
    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

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

    stage = args.stage
    torch.cuda.set_device(args.gpu)

    trg_num = domain_dict[args.trg_domain]
    src_num = domain_dict[args.src_domain]

    if (stage == 1):
        save_dir = join(save_root, args.save_dir, 'stage1')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print('domain: ', args.trg_domain)

        if(args.ssl):
            model = get_rot_model(args.model_name, num_domains=4)
            train_dataset = rot_dataset(args.data_root, 1, [args.trg_domain], 'train')
            val_dataset = rot_dataset(args.data_root, 1, [args.trg_domain], 'val')
        else:

            model = get_model(args.model_name, 65, 65, 4, pretrained=True)
            train_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], transform=train_transform)
            val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], transform=val_transform)

        train(args, model, train_dataset, val_dataset, stage, save_dir, trg_num)

        if (args.proceed):
            stage += 1

    if (stage == 2):
        save_dir = join(save_root, args.save_dir, 'stage2')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print('domain: ', args.src_domain)
        model = get_model(args.model_name, 65, 65, 4, pretrained=True)
        if (args.proceed):
            if(args.ssl):
                pre = torch.load(join(save_root, args.save_dir, 'stage1', 'best_model.ckpt'))['model']
                new_pre = OrderedDict()
                for p in pre:
                    if ('fc' in p):
                        continue
                    else:
                        new_pre[p] = pre[p]
                model.load_state_dict(new_pre, strict=False)
                del new_pre

            else:
                model.load_state_dict(torch.load(join(save_root, args.save_dir, 'stage1', 'best_model.ckpt'))['model'])
        else:
            model.load_state_dict(torch.load(join(save_root, args.model_path))['model'])

        bn_name = 'bns.'+(str)(src_num)
        for name, p in model.named_parameters():
            if ('fc' in name) or bn_name in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        torch.nn.init.xavier_uniform_(model.fc1.weight)
        torch.nn.init.xavier_uniform_(model.fc2.weight)
        train_dataset = OFFICEHOME_multi(args.data_root, 1, [args.src_domain], transform=train_transform)
        val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.src_domain], transform=val_transform)

        train(args, model, train_dataset, val_dataset, stage, save_dir, src_num)

        if (args.proceed):
            val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], transform=val_transform)
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

                    pred_val = model(x_val, trg_num * torch.ones_like(y_val), with_ft=False)

                    pred_vals.append(pred_val.cpu())

            pred_vals = torch.cat(pred_vals, 0)
            y_vals = torch.cat(y_vals, 0)
            total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])


if __name__ == '__main__':
    main()
