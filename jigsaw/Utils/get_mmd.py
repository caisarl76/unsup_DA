import numpy as np
import os
from os.path import join
import argparse

import torch
import torch.nn as nn
from torch.utils import data as util_data
from torch.autograd import Variable
from torchvision import transforms

import sys

sys.path.append('../Dataset')
sys.path.append('../Utils')

from dataset.datasets import MNIST, SVHN, SVHN_rotate
from model.lenet import DSBNLeNet
from jigsaw.network.AlexnetDSBN import AlexnetDSBN as Network
from model.resnetdsbn import resnet18dsbn
from jigsaw.Dataset.data_loader import DataLoader
from dataset.datasets import OFFICEHOME_multi, OFFICEHOME

domain_dict = {'RealWorld': 1, 'Clipart': 0, 'Product': 0, 'Art': 0}
new_dict = {'RealWorld': 1, 'Clipart': 0, 'Product': 3, 'Art': 2}

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

digit_dict = {'mnist': 0, 'svhn': 1}
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

svhn_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mmd(x, y, kernel):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        # print(source.shape)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # print(total0.shape, total1.shape)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # L2_distance = ((x - y) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        if (source.shape != target.shape):
            return -1
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # print(XX.shape, YX.shape, XY.shape, YY.shape)
        loss = torch.mean(XX + YY - XY - YX)
        return loss


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description='Train jigsaw puzzle archi with sup task')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder',
                        default="/data/jihun/OfficeHomeDataset_10072016/")
    parser.add_argument('--trg-domain', type=str, help='target domain of dataset', default=['Product'])
    parser.add_argument('--src-domain', type=str, help='source domain of dataset', default=['RealWorld'])
    parser.add_argument('--model-path',
                        default='/media/hd/jihun/dsbn_result/results/dsbn_ori/jigsaw_result/jigsaw_ssl_c_r/stage2/best_model.pth',
                        type=str, help='Path to pretrained model')
    parser.add_argument('--classes', default=[30, 65], type=int, help='Number of permutation to use', nargs='+')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
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


def main():
    args = parse_args()
    batch_size = 40

    # model = Network(classes=65, mmd=True)
    model = resnet18dsbn(pretrained=False, in_features=65, num_classes=65, num_domains=2)
    model.load_state_dict(torch.load(args.model_path)['model'])
    model.eval()
    model.cuda()
    # for name, p in model.named_parameters():
    #     print(name)
    model.layer3.register_forward_hook(model.printnorm)
    model.layer4.register_forward_hook(model.printnorm)



    t1 = OFFICEHOME_multi(args.data_path, num_domain=1, domain=args.trg_domain, transform=train_transform)
    t2 = OFFICEHOME_multi(args.data_path, num_domain=1, domain=args.trg_domain, transform=val_transform)
    s = OFFICEHOME_multi(args.data_path, num_domain=1, domain=args.src_domain, transform=train_transform)
    # t1 = DataLoader(join(args.data_path, args.trg_domain), split='train', classes=65, ssl=False)
    # t2 = DataLoader(join(args.data_path, args.trg_domain), split='val', classes=65, ssl=False)
    # s = DataLoader(join(args.data_path, args.src_domain), split='train', classes=65, ssl=False)

    t1_loader = torch.utils.data.DataLoader(dataset=t1, batch_size=batch_size, shuffle=True,
                                            num_workers=0)
    t2_loader = torch.utils.data.DataLoader(dataset=t2, batch_size=batch_size, shuffle=True,
                                            num_workers=0)
    s_loader = torch.utils.data.DataLoader(dataset=s, batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    t1_iter = enumerate(t1_loader)
    t2_iter = enumerate(t2_loader)
    s_iter = enumerate(s_loader)
    t1_num = domain_dict[args.trg_domain[0]]
    t2_num = domain_dict[args.trg_domain[0]]
    s_num = domain_dict[args.src_domain[0]]
    print(args.trg_domain[0], t1_num)
    print(args.trg_domain[0], t2_num)
    print(args.src_domain[0], s_num)

    mmd_loss = MMD_loss()

    try:
        _, (t1_x, t1_y) = t1_iter.__next__()
        _, (t2_x, t2_y) = t2_iter.__next__()
        _, (s_x, s_y) = s_iter.__next__()
    except StopIteration:
        t1_iter = enumerate(t1_loader)
        t2_iter = enumerate(t2_loader)
        s_iter = enumerate(s_loader)
        _, (t1_x, t1_y) = t1_iter.__next__()
        _, (t2_x, t2_y) = t2_iter.__next__()
        _, (s_x, s_y) = s_iter.__next__()

    t1_x, t2_x, s_x = t1_x.cuda(), t2_x.cuda(), s_x.cuda()
    # print(trg_x.shape)
    t1_pred = model(t1_x, t1_num * torch.ones(t1_x.shape[0], dtype=torch.long).cuda())
    t2_pred = model(t2_x, t2_num * torch.ones(t2_x.shape[0], dtype=torch.long).cuda())
    s_pred = model(s_x, s_num * torch.ones(s_x.shape[0], dtype=torch.long).cuda())

    homo_loss = mmd_loss(t1_pred, t2_pred)
    hete_loss = mmd_loss(t1_pred, s_pred)
    print(homo_loss, hete_loss)


# def main():
#     args = parse_args()
#     batch_size = 1024
#     model_path = '/media/hd/jihun/dsbn_result/results/dsbn_ori/digits-result/s_m_sup/stage2/best_model.ckpt'
#
#     model = DSBNLeNet(num_classes=10, weights_init_path=None, in_features=0, num_domains=2, mmd=True)
#     model.load_state_dict(torch.load(model_path)['model'])
#     model.bn2.register_forward_hook(model.printnorm)
#     model.eval()
#
#
#
#     svhn_train = SVHN(root='/data/jihun/SVHN', split='train', transform=svhn_transform, download=False)
#     svhn_test = SVHN(root='/data/jihun/SVHN', split='test', transform=svhn_transform, download=False)
#     mnist = MNIST('/data/jihun/MNIST', train=False, transform=mnist_transform, download=False)
#
#     svhn_train_loader = util_data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True,
#                                              num_workers=4, drop_last=True, pin_memory=True)
#     svhn_test_loader = util_data.DataLoader(svhn_test, batch_size=batch_size, shuffle=True,
#                                             num_workers=4, drop_last=True, pin_memory=True)
#
#     mnist_loader = util_data.DataLoader(mnist, batch_size=batch_size, shuffle=True,
#                                         num_workers=4, drop_last=True, pin_memory=True)
#
#     mmd_loss = MMD_loss()
#
#     # with torch.no_grad():
#     #     for (i, tr_x), (j, te_x) in zip(enumerate(svhn_train_loader), enumerate(svhn_test_loader)):
#     #         trx = tr_x[0].view(tr_x[0].size()[0], -1)
#     #         tex = te_x[0].view(te_x[0].size()[0], -1)
#     #         loss = mmd_loss(trx, tex)
#     #         print(loss)
#     #
#     #         break
#
#     with torch.no_grad():
#         for (_, m1), (_, s1), (_, s2) in zip(enumerate(mnist_loader), enumerate(svhn_train_loader),
#                                              enumerate(svhn_test_loader)):
#             m1_x = m1[0]
#             m1_x = m1_x.view(m1_x.size()[0], -1)
#
#             s1_x = s1[0]
#             s1_x = s1_x.view(s1_x.size()[0], -1)
#
#             s2_x = s2[0]
#             s2_x = s2_x.view(s2_x.size()[0], -1)
#
#             homo_loss = mmd_loss(s1_x, s2_x)
#             hete_loss = mmd_loss(s1_x, m1_x)
#             print(homo_loss, hete_loss)
#
#             m1_out = model(m1[0], 0 * torch.ones_like(m1[1]))
#             s1_out = model(s1[0], 1 * torch.ones_like(s1[1]))
#             s2_out = model(s2[0], 1 * torch.ones_like(s2[1]))
#
#             homo_out = mmd_loss(s1_out, s2_out)
#             hete_out = mmd_loss(s1_out, m1_out)
#             print(homo_out, hete_out)
#
#             break


if __name__ == '__main__':
    main()
