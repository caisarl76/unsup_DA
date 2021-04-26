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
from utils.train_utils import normal_train, test
from utils import io_utils, eval_utils

root = '/media/hd/jihun/dsbn_result/new/'

domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}
interval = [5000, 10000, 20000]

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Path to dataset folder',
                        default='/data/jihun/OfficeHomeDataset_10072016/')
    parser.add_argument('--save-root', help='directory to save models', default=None, type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='pseudo', type=str)
    # parser.add_argument('--teacher-model', help='dir where teacher model trained by src(stage1)', type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart')
    parser.add_argument('--src-domain', help='source training dataset', default='Clipart')

    parser.add_argument('--interval', help='experiment on stage1 intervals', action='store_true')
    parser.add_argument('--proceed', help='proceed to train student', action='store_true')
    parser.add_argument('--train-teacher', help='train teacher from scratch', action='store_true')
    parser.add_argument('--teacher-original', help='teacher model is DSBN original', action='store_true')
    parser.add_argument('--teacher-root', help='path where teacher model exists', type=str)


    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=10, type=int)
    parser.add_argument("--iters", type=int, default=[30050, 10050], help="choose gpu device.", nargs='+')
    # parser.add_argument("--iters", type=int, default=[2050, 550], help="choose gpu device.", nargs='+')
    # parser.add_argument("--iter", type=int, default=550, help="iteration for teacher training.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def ps_test(args, teacher, student, val_dataset, domain_num):
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

            # default number 1 for original dsbn implementation
            # 0:src 1: trg
            pseu_y = teacher(x_val, 1 * torch.ones_like(y_val), with_ft=False).argmax(axis=1)
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


def ps_train(args, teacher, student, train_dataset, val_dataset, save_dir, domain, iter):
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
    domain_num = domain_dict[domain]
    print('domain: %s , domain_num: %d' % (domain, domain_num))

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies

    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(iter):
        try:
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        except StopIteration:
            train_dataloader_iters = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iters.__next__()

        optimizer.zero_grad()

        x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)

        domain_idx = torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu)
        pred_y = student(x_s, domain_num * domain_idx, with_ft=False)
        # default number 1 for original dsbn implementation
        # 0:src 1: trg
        p_y = teacher(x_s, 1 * domain_idx, with_ft=False)
        # print(type(student), type(domain_num), type(domain_idx), type(x_s))

        loss = ce_loss(pred_y, p_y.argmax(axis=1))
        writer.add_scalar("Train Loss", loss, i)
        loss.backward()
        optimizer.step()

        if (i % 500 == 0 and i != 0):
            student, val_acc, student_acc = ps_test(args, teacher, student, val_dataset, domain_num)
            # print('%d iter || student acc: %0.3f, ||  val acc: %0.3f' % (i, student_acc, val_acc))
            writer.add_scalar("Val Accuracy", val_acc, i)
            writer.add_scalar("Student Accuracy", student_acc, i)

            model_dict = {'model': student.cpu().state_dict()}
            optimizer_dict = {'optimizer': optimizer.state_dict()}

            # save best checkpoint
            # io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

            if student_acc > best_accuracy:
                best_accuracy = student_acc
                model_dict = {'model': student.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=True)

            if (i % 10000 == 0 and i != 0):
                print('%d iter || student acc: %0.3f, ||  val acc: %0.3f' % (i, student_acc, val_acc))
            if args.interval and i in interval:
                print('%d iter student acc: %0.3f, || val acc: %0.3f' % (i, student_acc, val_acc))
                model_dict = {'model': student.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}
                # save best checkpoint
                io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

            student.train(True)
            student = student.cuda(args.gpu)

    writer.flush()
    writer.close()

    return student


def main():
    save_root = root
    args = parse_args()

    print('pseudo train, ')
    print('|%s| to |%s| training' % (args.src_domain, args.trg_domain))

    src_train = OFFICEHOME_multi(args.data_root, 1, [args.src_domain], split='train')
    src_val = OFFICEHOME_multi(args.data_root, 1, [args.src_domain], split='val')
    src_num = domain_dict[args.src_domain]

    trg_train = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='train')
    trg_val = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain], split='val')
    trg_num = domain_dict[args.trg_domain]

    ###################### train teacher model ######################
    if (args.save_root):
        save_root = args.save_root
    torch.cuda.set_device(args.gpu)

    t_path = join(args.teacher_root, '%s_%s/' % (args.src_domain, args.trg_domain))

    # teacher for our custum dsbn
    teacher = get_model(args.model_name, num_classes=65, in_features=65, num_domains=4, pretrained=True)
    t2_path = join(t_path, 'teacher/stage2/best_model.ckpt')

    if (args.teacher_original):
        # teacher for original dsbn
        teacher = get_model(args.model_name, num_classes=65, in_features=0, num_domains=2, pretrained=True)
        t2_path = join(t_path, 'stage2/best_resnet50dsbn+None+i0_%s2%s.pth' % (args.src_domain, args.trg_domain))

    print(t2_path)
    # if not os.path.isfile(t2_path) or args.train_teacher:
    if not os.path.isfile(t2_path):
        print('train teacher2 ')

        t1_path = join(t_path, 'stage1/best_model.ckpt')
        if os.path.isfile(t1_path):
            print('teacher1 exists: ', t1_path)
            teacher.load_state_dict(torch.load(t1_path)['model'])
        else:
            print('train teacher1')
            save_dir = join(save_root, args.save_dir, 'teacher/stage1')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            print('save dir: ', save_dir)
            teacher = normal_train(args, teacher, src_train, src_val, args.iters[0], save_dir, args.src_domain)

        for name, p in teacher.named_parameters():
            p.requires_grad = False

        save_dir = join(save_root, args.save_dir, 'teacher/stage2')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print('save dir: ', save_dir)

        normal_train(args, teacher, trg_train, trg_val, args.iters[1], save_dir, args.trg_domain, freeze=True)

    else:
        print('teacher2 exists: ', t2_path)
        teacher.load_state_dict(torch.load(t2_path)['model'])

    ###################### train student model ######################
    #################################### STAGE 1 ####################################
    save_dir = join(save_root, args.save_dir, 'student/stage1')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print('save dir: ', save_dir)

    for name, p in teacher.named_parameters():
        p.requires_grad = False

    student = get_model(args.model_name, 65, 65, 4, pretrained=True)
    student = ps_train(args, teacher, student, trg_train, trg_val, save_dir, args.trg_domain, args.iters[0])

    #################################### STAGE 2 ####################################

    weight_pth = {}
    if(args.interval):
        for i in interval:
            weight_pth[i] = join(save_dir, (str)(i) + '_weight.ckpt')
        weight_pth['best'] = join(save_dir, 'best_model.ckpt')

        for i in weight_pth.keys():
            save_dir = join(save_root, args.save_dir, 'student/stage2/', (str)(i))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            print('save dir: ', save_dir)
            student.load_state_dict(torch.load(weight_pth[i])['model'])

            bn_name = 'bns.' + (str)(src_num)
            for name, p in student.named_parameters():
                if ('fc' in name) or bn_name in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            student = normal_train(args, student, src_train, src_val, args.iters[1], save_dir, args.src_domain)

            #################################### STAGE 3 ####################################

            _, stage3_acc = test(args, student, trg_val, trg_num)
            print('####################################')
            print('### stage 3 at stage1 iter:', i, '||  %0.3f' % (stage3_acc))
            print('####################################')

    save_dir = join(save_root, args.save_dir, 'student/stage2')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print('save dir: ', save_dir)

    src_bn = 'bns.' + (str)(src_num)
    trg_bn = 'bns.' + (str)(trg_num)

    weight_dict = OrderedDict()
    for name, p in student.named_parameters():
        if (trg_bn in name):
            weight_dict[name] = p
            new_name = name.replace(trg_bn, src_bn)
            weight_dict[new_name] = p
        elif(src_bn in name):
            continue
        else:
            weight_dict[name] = p

    student.load_state_dict(weight_dict, strict=False)
    del weight_dict

    for name, p in student.named_parameters():
        if ('fc' in name):
            p.requires_grad = True
        else:
            p.requires_grad = False

    student = normal_train(args, student, src_train, src_val, args.iters[1], save_dir, args.src_domain)

    #################################### STAGE 3 ####################################

    _, stage3_acc = test(args, student, trg_val, trg_num)
    print('####################################')
    print('### stage 3 at stage1 iter: best', '||  %0.3f' % (stage3_acc))
    print('####################################')

if __name__ == '__main__':
    main()
