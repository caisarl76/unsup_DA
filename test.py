import argparse
from collections import OrderedDict
from os.path import join as join
import torch
from torch.utils import data as util_data
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import MNIST, SVHN
from rot_dataset.rot_dataset import rot_dataset
from dataset.datasets import OFFICEHOME_multi
from dataset.factory import get_dataset
from model.factory import get_model
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

domain_dict = {'RealWorld': 1, 'Clipart': 0, 'Product': 0, 'Art':0}
domain_dict = {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3}
# save_root = '/media/hd/jihun/dsbn_result/results/dsbn_ori/'
save_root = '/media/hd/jihun/dsbn_result/new/'
# save_root = '.'
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

svhn_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(),
    transforms.ToTensor()
])

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
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/jihun/OfficeHomeDataset_10072016', type=str)
    parser.add_argument('--ssl', help='is it a ssl model', default=0, type=int)
    parser.add_argument('--save-dir', help='directory to save models', default='result/try1', type=str)
    parser.add_argument('--model-path', help='directory to save models', default='result/try1/best_model.ckpt',
                        type=str)
    parser.add_argument('--model-name', help='model name', default='resnet50dsbn')
    parser.add_argument('--domain', help='source training dataset', default=['Clipart'], nargs='+')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[10000, 10000], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    num_domains = 1
    domain_num = domain_dict[args.domain[0]]
    # print('domain num:', domain_num)
    # print(args.ssl)

    model = get_model(args.model_name, 65, 65, 4, pretrained=False)

    if (args.ssl == 1):
        pre = torch.load(join(save_root, args.model_path))['model']
        new_pre = OrderedDict()

        for p in pre:
            if ('fc' in p):
                continue
            else:
                new_pre[p] = pre[p]

        model.load_state_dict(new_pre, strict=False)
        del new_pre
    else:
        model.load_state_dict(torch.load(join(save_root, args.model_path))['model'])
        # model.load_state_dict(torch.load(args.model_path)['model'])

    model.eval()
    model = model.cuda(args.gpu)
    test_dataset = OFFICEHOME_multi(args.data_root, num_domains, args.domain, split='val')
    # test_dataset = rot_dataset(args.data_root, num_domains, args.domain, 'val')

    # test_dataset = SVHN(root='/data/jihun/SVHN', split='test', transform=svhn_transform, download=True)
    # test_dataset = MNIST('/data/jihun/MNIST', train=False, transform=mnist_transform, download=True)
    test_dataloader = util_data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_dataloader_iter = enumerate(test_dataloader)

    pred_vals = []
    y_vals = []

    with torch.no_grad():
        for j, (x_val, y_val) in test_dataloader_iter:
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
                           for c, c_name in enumerate(test_dataset.classes)]

    mean_val_accuracy = float(
        torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))

    print('%0.3f'%(total_val_accuracy))
    # for cls in val_accuracy_each_c:
    #     print(cls)
    # print(mean_val_accuracy)
    del x_val, y_val, pred_val, pred_vals, y_vals
    del test_dataloader_iter

    return


if __name__ == '__main__':
    main()
