import argparse
import os
from os.path import join as join
import torch
from collections import OrderedDict
from model.rot_resnetdsbn import get_rot_model
from model.factory import get_model
from utils.train_utils import normal_train, test

from dataset.get_dataset import get_dataset

# domain_dict = {'RealWorld': 1, 'Clipart': 0}
root = '/media/hd/jihun/dsbn_result/'

data_pth_dict = {'officehome': 'OfficeHomeDataset_10072016', 'domainnet': 'domainnet'}
domain_dict = {'officehome': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='directory where dataset exists',
                        default='officehome', type=str)
    parser.add_argument('--data-root', help='directory where dataset exists',
                        default='/data/', type=str)
    parser.add_argument('--save-root', help='directory to save models', default='/results/result/rot_ssl/', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='domainnet/clipart_sketch', type=str)
    parser.add_argument('--model-name', default='resnet50dsbn', type=str)
    parser.add_argument('--model-path', default='path for stage2 train', type=str)
    parser.add_argument('--trg-domain', help='target training dataset', default='Clipart')
    parser.add_argument('--src-domain', help='target training dataset', default='RealWorld')

    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--iters", type=int, default=[30005, 10005], help="choose gpu device.", nargs='+')
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--lr-scheduler', '-lrsche', dest='lr_scheduler',
                        help='learning_rate scheduler [Lambda/Multiplicate/Step/Multistep/Expo', type=str)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    parser.add_argument("--ssl", action='store_true')
    parser.add_argument("--only1", action='store_true')
    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    save_root = root
    stage = args.stage

    num_domain = 4
    num_classes = 65

    if args.dataset == 'domainnet':
        num_domain = 6
        num_classes = 345
    elif args.dataset == 'officehome':
        num_domain = 4
        num_classes = 65

    if (args.save_root):
        save_root = args.save_root

    trg_sup_train, trg_sup_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root, domain=args.trg_domain,
                                             ssl=False)
    trg_num = domain_dict[args.dataset][args.trg_domain]
    src_train, src_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root, domain=args.src_domain,
                                     ssl=False)
    src_num = domain_dict[args.dataset][args.src_domain]

    save_dir = None
    model = None

    #################################### STAGE 1 ####################################
    if stage == 1:
        if args.ssl:
            save_dir = join(save_root, 'stage1/rot/', args.trg_domain)
            if not os.path.isfile(join(save_dir, 'best_model.ckpt')):

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                model = get_model(args.model_name, in_features=256, num_classes=4, num_domains=num_domain,
                                  pretrained=True)
                trg_ssl_train, trg_ssl_val = get_dataset(dataset=args.dataset, dataset_root=args.data_root,
                                                         # domain=args.trg_domain,
                                                         domain=[args.trg_domain, args.src_domain],
                                                         ssl=True)
                print('train stage 1')
                model = normal_train(args, model, trg_ssl_train, trg_ssl_val, args.iters[0], save_dir, args.trg_domain)
            else:
                print('find stage 1 model: ', save_dir)
        else:
            save_dir = join(save_root, 'stage1/sup/', args.trg_domain)
            if not os.path.isfile(join(save_dir, 'best_model.ckpt')):
                print('train stage 1')
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                model = get_model(args.model_name, in_features=num_classes, num_classes=num_classes,
                                  num_domains=num_domain, pretrained=True)

                model = normal_train(args, model, trg_sup_train, trg_sup_val, args.iters[0], save_dir, args.trg_domain)
            else:
                print('find stage 1 model: ', save_dir)
        if args.only1:
            stage = 1
        else:
            stage += 1

    #################################### STAGE 2 ####################################
    if stage == 2:
        print('train stage 2')
        if args.ssl:
            model_pth = join(save_root, 'stage1/rot/', args.trg_domain, 'best_model.ckpt')
            print('load model from %s' % (model_pth))
            pre = torch.load(model_pth)
            save_dir = join(save_root, 'stage2/rot', args.save_dir)
        else:
            model_pth = join(save_root, 'stage1/sup/', args.trg_domain, 'best_model.ckpt')
            print('load model from %s' % (model_pth))
            pre = torch.load(model_pth)
            save_dir = join(save_root, 'stage2/sup', args.save_dir)

        model = get_model(args.model_name, in_features=num_classes, num_classes=num_classes,
                          num_domains=num_domain, pretrained=False)
        model.load_state_dict(pre, strict=False)

        src_bn = 'bns.' + (str)(src_num)
        trg_bn = 'bns.' + (str)(trg_num)

        weight_dict = OrderedDict()
        for name, p in model.named_parameters():
            if (trg_bn in name):
                weight_dict[name] = p
                new_name = name.replace(trg_bn, src_bn)
                weight_dict[new_name] = p
            elif (src_bn in name):
                continue
            else:
                weight_dict[name] = p
        model.load_state_dict(weight_dict, strict=False)
        for name, p in model.named_parameters():
            p.requires_grad = False

        model.fc1.weight.requires_grad = True
        model.fc2.weight.requires_grad = True
        torch.nn.init.xavier_uniform_(model.fc1.weight)
        torch.nn.init.xavier_uniform_(model.fc2.weight)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        model = normal_train(args, model, src_train, src_val, args.iters[1], save_dir, args.src_domain,
                             test_datset=trg_sup_val, test_domain=args.trg_domain)
    #################################### STAGE 3 ####################################
    _, stage3_acc = test(args, model, trg_sup_val, domain_dict[args.dataset][args.trg_domain])
    print('####################################')
    print('### stage 3 at stage1 iter: %0.3f' % (stage3_acc))
    print('####################################')


if __name__ == '__main__':
    main()
