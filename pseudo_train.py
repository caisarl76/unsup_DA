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

root = '/media/hd/jihun/dsbn_result/new/'

domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--save-root', help='directory to save models', default=None, type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='result/try1', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/try1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--trg_domain', help='target training dataset', default='Clipart')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=10, type=int)
    parser.add_argument("--iter", type=int, default=550, help="choose gpu device.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def test(args, teacher, student, val_dataset, save_dir, domain_num):
    val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_dataloader_iter = enumerate(val_dataloader)

    val_accs_each_c = []
    student_accs_each_c = []

    pseu_ys = []
    pred_ys = []
    y_vals = []
    x_val = None
    y_val = None

    teacher.eval()
    student.eval()

    with torch.no_grad():
        for j, (x_val, y_val) in val_dataloader_iter:
            y_vals.append(y_val.cpu())
            x_val = x_val.cuda(args.gpu)
            y_val = y_val.cuda(args.gpu)

            pseu_y = teacher(x_val, domain_num * torch.ones_like(y_val), with_ft=False).argmax(axis=1)
            pred_y = student(x_val, domain_num * torch.ones_like(y_val), with_ft=False)
            pseu_ys.append(pseu_y.cpu())
            pred_ys.append(pred_y.cpu())

    pred_ys = torch.cat(pred_ys, 0)
    pseu_ys = torch.cat(pseu_ys, 0)
    y_vals = torch.cat(y_vals, 0)

    val_acc = float(eval_utils.accuracy(pred_ys, y_vals, topk=(1,))[0])
    val_acc_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_ys, y_vals,
                                                              class_idx=c, topk=(1,))[0]))
                      for c, c_name in enumerate(val_dataset.classes)]
    student_acc = float(eval_utils.accuracy(pred_ys, pseu_ys, topk=(1,))[0])
    student_acc_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_ys, pseu_ys,
                                                                  class_idx=c, topk=(1,))[0]))
                          for c, c_name in enumerate(val_dataset.classes)]
    val_accs_each_c.append(val_acc_each_c)
    student_accs_each_c.append(student_acc_each_c)

    del x_val, y_val, pred_y, pred_ys, pseu_y, pseu_ys, y_vals
    del val_dataloader_iter

    return student, val_acc, student_acc


def train(args, teacher, student, train_dataset, val_dataset, save_dir, domain_num):
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    train_dataloader_iters = enumerate(train_dataloader)

    teacher.eval()
    student.train(True)
    teacher = teacher.cuda(args.gpu)
    student = student.cuda(args.gpu)

    params = get_optimizer_params(student, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=True, base_weight_factor=0.1)

    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    ce_loss = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=join(save_dir, 'logs'))
    print('domain: %s , domain_num: %d' % (args.trg_domain, domain_num))

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
        pred_y = student(x_s, domain_num * domain_idx, with_ft=False)
        p_y = teacher(x_s, domain_num * domain_idx, with_ft=False)
        # print(type(student), type(domain_num), type(domain_idx), type(x_s))


        loss = ce_loss(pred_y, p_y.argmax(axis=1))
        writer.add_scalar("Train Loss", loss, i)
        loss.backward()
        optimizer.step()

        if (i % 500 == 0 and i != 0):
            student, val_acc, student_acc = test(args, teacher, student, val_dataset, save_dir, domain_num)
            print('%d iter student acc: %0.3f,   val acc: %0.3f' % (i, student_acc, val_acc))
            writer.add_scalar("Val Accuracy", val_acc, i)
            writer.add_scalar("Student Accuracy", student_acc, i)

            model_dict = {'model': student.cpu().state_dict()}
            optimizer_dict = {'optimizer': optimizer.state_dict()}

            # save best checkpoint
            io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

            if student_acc > best_accuracy:
                best_accuracy = student_acc
                model_dict = {'model': student.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=True)

            if (i % 10000 == 0 and i != 0):
                print('%d iter complete' % (i))

            student.train(True)
            student = student.cuda(args.gpu)

    writer.flush()
    writer.close()

    return


def main():
    save_root = root
    args = parse_args()
    if (args.save_root):
        save_root = args.save_root
        print('save root: ', save_root)

    torch.cuda.set_device(args.gpu)

    save_dir = join(save_root, args.save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print('domain: ', args.trg_domain)

    student = get_model(args.model_name, 65, 65, 4, pretrained=True)
    teacher = get_model(args.model_name, 65, 65, 4, pretrained=True)
    # teacher.load_state_dict(torch.load(args.model_path))

    train_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='train')
    val_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='val')

    train(args, teacher, student, train_dataset, val_dataset, save_dir, domain_dict[args.trg_domain])


if __name__ == '__main__':
    main()
