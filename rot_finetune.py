import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.factory import get_dataset
from rot_dataset.rot_dataset import rot_dataset
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils
from collections import OrderedDict

domain_dict = {'RealWorld': 1, 'Clipart': 0}
save_root = '/media/hd/jihun/dsbn_result/'


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/OfficeHomeDataset_10072016', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='ssl_result/0315_3/', type=str)
    parser.add_argument('--model-path', help='directory to save models',
                        default='ssl_result/0315_3/stage1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet18dsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='RealWorld')
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart', nargs='+')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=10000, help="choose gpu device.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    stage = args.stage


    save_dir = join(save_root, args.save_dir, 'stage1')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('stage: %d  , domain: %s' % (stage, args.trg_domain))

    model = get_model(args.model_name, 65, 65, 4)
    # if (args.proceed) and (args.model_path):
    #     pre = torch.load(join(save_root, args.save_dir, 'stage1', 'best_model.ckpt'))['model']
    # elif (not args.proceed) and (args.model_path):
    #     pre = torch.load(join(save_root, args.model_path))['model']
    pre = torch.load(join(save_root, args.model_path))['model']
    new_pre = OrderedDict()

    for p in pre:
        if ('fc' in p):
            continue
        else:
            new_pre[p] = pre[p]

    model.load_state_dict(new_pre, strict=False)
    del new_pre

    for p in model.parameters():
        p.requires_grad = False

    model.fc1.weight.requires_grad = True
    model.fc2.weight.requires_grad = True
    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)

    train_dataset = get_dataset("{}_{}_{}_{}".format(args.model_name, args.trg_domain, 'train', None))
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    train_dataloader_iters = enumerate(train_dataloader)

    val_dataset = get_dataset("{}_{}_{}_{}".format(args.model_name, args.trg_domain, 'val', None))

    model.train(True)
    model = model.cuda(args.gpu)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=True, base_weight_factor=0.1)
    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    ce_loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    domain_num = domain_dict[train_dataset.domain]
    print('domain_num, stage: ', domain_num, stage)

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies
    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(args.iters):
        try:
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        except StopIteration:
            train_dataloader_iters = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iters.__next__()

        optimizer.zero_grad()

        x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
        pred, f = model(x_s, domain_num * torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu), with_ft=True)
        loss = ce_loss(pred, y_s)
        writer.add_scalar("Train Loss", loss, i)
        loss.backward()
        optimizer.step()

        if (i % 500 == 0 and i != 0):
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

        if (i % 10000 == 0 and i != 0):
            print('%d iter complete' % (i))
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
