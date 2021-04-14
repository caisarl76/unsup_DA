import argparse
import torch.nn.functional as F
import torch.optim as optim
import os
from os.path import join as join
import torch
from torch.utils import data as util_data
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rot_dataset.rot_dataset import rot_dataset
from model.rot_resnetdsbn import get_rot_model

from dataset.datasets import OFFICEHOME_multi
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

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
    parser.add_argument('--data-root', help='directory data exists', default='/data/jihun/OfficeHomeDataset_10072016',
                        type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='result/try1', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/try1/best_model.ckpt',
                        type=str)
    parser.add_argument('--ssl', help='exp set on ssl', action='store_true')
    parser.add_argument('--model-name', help='model name', default='resnet18dsbn')
    parser.add_argument('--src-domain', help='source training dataset', default='RealWorld')
    parser.add_argument('--trg-domain', help='target training dataset', default=['Clipart', 'Product'], nargs='+')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[30000, 10000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=2e-3, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument('--proceed', help='proceed to next stage', default=True, type=bool)
    parser.add_argument('--stage', help='starting stage', default=1, type=int)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def test(args, dataset, model):
    dataloader = util_data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dataloader_iter = enumerate(dataloader)
    model.eval()
    pred_vals = []
    y_vals = []
    x_val = None
    y_val = None
    # print('------------------------dataload------------------------')
    with torch.no_grad():
        for j, (x_val, y_val) in dataloader_iter:
            y_vals.append(y_val.cpu())
            x_val = x_val.cuda(args.gpu)
            y_val = y_val.cuda(args.gpu)

            pred_val = model(x_val, 0 * torch.ones_like(y_val), with_ft=False)

            pred_vals.append(pred_val.cpu())

    pred_vals = torch.cat(pred_vals, 0)
    y_vals = torch.cat(y_vals, 0)
    total_test_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
    # test2_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
    #                                                                  class_idx=c, topk=(1,))[0]))
    #                          for c, c_name in enumerate(dataset.classes)]

    print('Test accuracy for domain: ', dataset.domain)
    print('mean acc: ', total_test_accuracy)
    # for item in test2_accuracy_each_c:
    #     print(item[0], item[1])

    return


def main():
    args = parse_args()
    stage = args.stage
    torch.cuda.set_device(args.gpu)
    writer = SummaryWriter()

    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('domain: ', args.trg_domain)

    num_domain = len(args.trg_domain)

    if (args.ssl):
        model = get_rot_model(args.model_name, num_domains=1)
        train_dataset = rot_dataset(args.data_root, num_domain, args.trg_domain, 'train')
        val_dataset = rot_dataset(args.data_root, num_domain, args.trg_domain, 'val')
        test1_dataset = rot_dataset(args.data_root, 1, [args.trg_domain[0]], 'test')
        if (len(args.trg_domain) > 1):
            test2_dataset = rot_dataset(args.data_root, 1, [args.trg_domain[1]], 'test')

    else:
        model = get_model(args.model_name, 65, 65, 1, pretrained=True)
        train_dataset = OFFICEHOME_multi(args.data_root, num_domain, args.trg_domain, transform=train_transform)
        val_dataset = OFFICEHOME_multi(args.data_root, num_domain, args.trg_domain, transform=val_transform)
        test1_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain[0]], transform=val_transform)
        if (len(args.trg_domain) > 1):
            test2_dataset = OFFICEHOME_multi(args.data_root, 1, [args.trg_domain[1]], transform=val_transform)

    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)

    train_dataloader_iter = enumerate(train_dataloader)

    model.train(True)
    model = model.cuda(args.gpu)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies
    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(args.iters[0]):
        try:
            _, (x_s, y_s) = train_dataloader_iter.__next__()
        except StopIteration:
            train_dataloader_iter = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iter.__next__()
        optimizer.zero_grad()

        x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
        domain_idx = torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu)
        pred, f = model(x_s, 0 * domain_idx, with_ft=True)
        loss = ce_loss(pred, y_s)
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

                    pred_val = model(x_val, 0 * torch.ones_like(y_val), with_ft=False)

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
            # io_utils.save_check(save_dir, i, model_dict, optimizer_dict, best=False)

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

        if (i % 5000 == 0 and i != 0):
            print('%d iter complete' % (i))
            test(args, test1_dataset, model)
            if (len(args.trg_domain)> 1):
                test(args, test2_dataset, model)

    writer.flush()
    writer.close()

    model.eval()
    test(args, test1_dataset, model)
    if (len(args.trg_domain) > 1):
        test(args, test2_dataset, model)


if __name__ == '__main__':
    main()
