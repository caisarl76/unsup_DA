from model import resnetdsbn
import argparse
import logging
import pprint
import datetime
import sys
import random

from dataset.factory import get_dataset
from model.factory import get_model
from discriminator.factory import get_discriminator

from model.centroids import Centroids
from torch.utils import data as util_data
from utils.train_utils import adaptation_factor, semantic_loss_calc, get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils
from dataset.datasets import MNIST, SVHN

import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import os
import torch
import torch.nn as nn
import numpy as np

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

svhn_transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(),
    transforms.ToTensor()
])


def parse_args(args=None, namespace=None):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Finetune DSBN Network.\n' + \
                    'target label:0, sorce label:1,2,... \n' + \
                    'office-home: Art, Clipart, Product, RealWorld')
    parser.add_argument('--model-name',
                        help="model name ['resnet18dsbn']",
                        default='resnet18dsbn', type=str)
    parser.add_argument('--exp-setting', help='exp setting[digits, office, imageclef, visda]', default='office-home',
                        type=str)
    parser.add_argument('--teacher-model-path', help='teacher model path', default='', type=str)
    parser.add_argument('--init-model-path', help='init model path', default='', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='output/try2', type=str)

    # model options
    parser.add_argument('--num-classes', help='number of classes', default=0, type=int)

    parser.add_argument('--finetune-dataset', help='source training dataset', default='RealWorld')
    # parser.add_argument('--source-datasets', help='source training dataset', default=['amazon', 'dslr'],
    #                     nargs='+')
    # parser.add_argument('--merge-sources', help='Use merged dataset as source dataset.', action='store_true')
    # parser.add_argument('--target-datasets', help='target training dataset', default=['webcam'], nargs='+')
    parser.add_argument('--in-features', help='add in feature dimension. 0 for label logit space.', default=0,
                        type=int)
    parser.add_argument('--jitter', default='None', type=str,
                        help='data loader additional jitter type. None option is default jittering(rgb224)' +
                             ' [None, grey224, grey160, rgb160]')

    # machine options
    parser.add_argument('--num-workers', help='number of worker to load data', default=5, type=int)
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument('--manual-seed', type=int, default=0, help='manual random seed')

    # train hyper-parameters
    parser.add_argument('--max-step', help='maximum step', default=30000, type=int)
    parser.add_argument('--early-stop-step', help='early stop step', default=30000, type=int)
    parser.add_argument('--warmup-learning-rate', '-wlr', help='warmup learning rate', default=0.0, type=float)
    parser.add_argument('--warmup-step', type=int, default=0, help='warm-up iterations')
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-3, type=float)
    parser.add_argument('--beta1', dest='beta1', help='beta1 for Adam', default=0.9, type=float)
    parser.add_argument('--beta2', dest='beta2', help='beta2 for Adam', default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--double-bias-lr', help='double-bias', action='store_true')
    parser.add_argument('--base-weight-factor', help='reduce base_weight learning rate by the factor value',
                        default=0.1, type=float)
    parser.add_argument('--sm-etha', help='sm loss adjust factor', default=1.0, type=float)
    parser.add_argument('--pseudo-target-threshold', help='threshold for calculating pseudo-t loss', default=0.0,
                        type=float)
    parser.add_argument('--no-lambda', help='double-bias', action='store_true')
    parser.add_argument('--optimizer', help='[Adam/SGD]', default='Adam', type=str)

    # trainval parameters
    parser.add_argument('--adaptation-gamma', help='adaptation gamma value', default=10, type=float)
    parser.add_argument('--domain-loss-adjust-factor', help='domain loss factor', default=0.1, type=float)
    parser.add_argument('--adv-loss', help='add domain loss', action='store_true')
    parser.add_argument('--sm-loss', help='add moving semantic loss', action='store_true')
    parser.add_argument('--pseudo-target-loss',
                        help='target classification loss with pseudo label, [default/score]_[ensemble]_[fix]',
                        default='', type=str)

    # log and diaplay
    parser.add_argument('--use-tfboard', help='whether use tensorflow tensorboard',
                        action='store_true')
    parser.add_argument('--save-model-hist', help='save model histogram on tfboard', action='store_true')
    parser.add_argument('--disp-interval', help='number of iterations to display', default=10,
                        type=int)
    parser.add_argument('--save-interval',
                        help='number of iterations to save. if save_interval < 0, no saving mode.',
                        default=500,
                        type=int)
    parser.add_argument('--print-console', help='activate console display', action='store_true')
    parser.add_argument('--save-ckpts', help='whether to save the checkpoints in every save_interval',
                        action='store_true')
    parser.add_argument('--resume', help='resume from latest(or best) checkpoint', action='store_true')

    args = parser.parse_args(args=args, namespace=namespace)
    return args


def main():
    print('start finetune')
    args = parse_args()
    args.dsbn = True if 'dsbn' in args.model_name else False  # set dsbn
    args.cpua = True if 'cpua' in args.model_name else False

    torch.cuda.set_device(args.gpu)  # set current gpu device id so pin_momory works on the target gpu
    start_time = datetime.datetime.now()  # execution start time

    # make save_dir
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # check whether teacher model exists
    if not os.path.isfile(args.teacher_model_path):
        raise AttributeError('Missing teacher model path: {}'.format(args.teacher_model_path))

    # create log file
    log_filename = 'train_records.log'
    log_path = os.path.join(args.save_dir, log_filename)
    logger = io_utils.get_logger(__name__, log_file=log_path, write_level=logging.INFO,
                                 print_level=logging.INFO if args.print_console else None,
                                 mode='a' if args.resume else 'w')

    # set num_classes by checking exp_setting
    if args.num_classes == 0:
        if args.exp_setting in ['office-home']:
            logger.warning('num_classes are not 65! set to 65.')
            args.num_classes = 65
        elif args.exp_setting in ['digits']:
            logger.warning('num_classes are not 10! set to 10.')
            args.num_classes = 10
        else:
            raise AttributeError('Wrong num_classes: {}'.format(args.num_classes))

    if args.manual_seed:
        # set manual seed
        args.manual_seed = np.uint32(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        logger.info('Random Seed: {}'.format(int(args.manual_seed)))
        args.random_seed = args.manual_seed  # save seed into args
    else:
        seed = np.uint32(random.randrange(sys.maxsize))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(np.uint32(seed))
        logger.info('Random Seed: {}'.format(seed))
        args.random_seed = seed  # save seed into args

    if args.resume:
        logger.info('Resume training')
    else:
        logger.info('\nArguments:\n' + pprint.pformat(vars(args), indent=4))  # print args
    torch.save(vars(args), os.path.join(args.save_dir, 'args_dict.pth'))  # save args

    num_classes = args.num_classes
    in_features = args.in_features if args.in_features != 0 else num_classes
    # num_domains = len(args.source_datasets) + len(args.target_datasets)

    # tfboard
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        tfboard_dir = os.path.join(args.save_dir, 'tfboard')
        if not os.path.isdir(tfboard_dir):
            os.makedirs(tfboard_dir)
        writer = SummaryWriter(tfboard_dir)

    # resume
    if args.resume:
        try:
            checkpoints = io_utils.load_latest_checkpoints(args.save_dir, args, logger)
        except FileNotFoundError:
            logger.warning('Latest checkpoints are not found! Trying to load best model...')
            checkpoints = io_utils.load_best_checkpoints(args.save_dir, args, logger)

        start_iter = checkpoints[0]['iteration'] + 1
    else:
        start_iter = 1

    ###################################################################################################################
    #                                               Data Loading                                                      #
    ###################################################################################################################

    # train_dataset = MNIST('/data/jihun/MNIST', train=True, transform=mnist_transform, download=True)
    # val_dataset = MNIST('/data/jihun/MNIST', train=False, transform=mnist_transform, download=True)
    train_dataset = SVHN(root='/data/jihun/SVHN', transform=svhn_transform, download=True)
    val_dataset = SVHN(root='/data/jihun/SVHN', split='test', transform=svhn_transform, download=True)
    train_dataloader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_dataloader = util_data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)

    train_dataloader_iters = enumerate(train_dataloader)
    val_dataloader_iter = enumerate(val_dataloader)

    ###################################################################################################################
    #                                               Model Loading                                                     #
    ###################################################################################################################
    model = get_model(args.model_name, 10, 0, 2, pretrained=True)

    params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,
                                  double_bias_lr=args.double_bias_lr, base_weight_factor=args.base_weight_factor)

    # teacher model
    # print(teacher_model)
    print('------------------------model load------------------------')
    model.load_state_dict(torch.load(args.teacher_model_path)['model'])
    for name, p in model.state_dict().items():
        if ('fc' in name) or 'bns.1' in name:
            continue
        else:
            p.requires_grad = False

    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)

    model.train(True)
    model = model.cuda(args.gpu)

    ###################################################################################################################
    #                                               Train Configurations                                              #
    ###################################################################################################################
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params, betas=(args.beta1, args.beta2))

    # Train Starts
    logger.info('Train Starts')
    monitor = Monitor()

    global best_accuracy
    global best_accuracies_each_c
    global best_mean_val_accuracies
    global best_total_val_accuracies
    best_accuracy = 0.0
    best_accuracies_each_c = []
    best_mean_val_accuracies = []
    best_total_val_accuracies = []

    for i in range(start_iter, args.early_stop_step + 1):
        try:
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        except StopIteration:
            train_dataloader_iters = enumerate(train_dataloader)
            _, (x_s, y_s) = train_dataloader_iters.__next__()
        # init optimizer
        optimizer.zero_grad()
        # ship to cuda
        x_s, y_s = x_s.cuda(args.gpu), y_s.cuda(args.gpu)
        pred_s, f_s = model(x_s, 1 * torch.ones(x_s.shape[0], dtype=torch.long).cuda(args.gpu),
                            with_ft=True)
        loss = ce_loss(pred_s, y_s)
        monitor.update({"Loss/Closs_src": float(loss)})

        loss.backward()
        optimizer.step()

        if i % args.save_interval == 0 and i != 0:
            # print('------------------------%d val start------------------------' % (i))
            logger.info("Elapsed Time: {}".format(datetime.datetime.now() - start_time))
            logger.info("Start Evaluation at {:d}".format(i))

            total_val_accuracies = []
            mean_val_accuracies = []
            val_accuracies_each_c = []
            model.eval()

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

                    pred_val = model(x_val, 1 * torch.ones_like(y_val), with_ft=False)

                    pred_vals.append(pred_val.cpu())
            # print('------------------------acc compute------------------------')
            pred_vals = torch.cat(pred_vals, 0)
            y_vals = torch.cat(y_vals, 0)
            total_val_accuracy = float(eval_utils.accuracy(pred_vals, y_vals, topk=(1,))[0])
            val_accuracy_each_c = [(c_name, float(eval_utils.accuracy_of_c(pred_vals, y_vals,
                                                                           class_idx=c, topk=(1,))[0]))
                                   for c, c_name in enumerate(val_dataset.classes)]

            logger.info('\n{} Accuracy of Each class\n'.format(args.finetune_dataset) +
                        ''.join(["{:<25}: {:.2f}%\n".format(c_name, 100 * c_val_acc)
                                 for c_name, c_val_acc in val_accuracy_each_c]))
            mean_val_accuracy = float(
                torch.mean(torch.FloatTensor([c_val_acc for _, c_val_acc in val_accuracy_each_c])))
            # print('------------------------mean acc------------------------')
            logger.info('{} mean Accuracy: {:.2f}%'.format(
                args.finetune_dataset, 100 * mean_val_accuracy))
            logger.info(
                '{} Accuracy: {:.2f}%'.format(args.finetune_dataset,
                                              total_val_accuracy * 100))

            total_val_accuracies.append(total_val_accuracy)
            val_accuracies_each_c.append(val_accuracy_each_c)
            mean_val_accuracies.append(mean_val_accuracy)
            # print('------------------------tf board------------------------')
            if args.use_tfboard:
                writer.add_scalar('Val_acc', total_val_accuracy, i)
                for c_name, c_val_acc in val_accuracy_each_c:
                    writer.add_scalar('Val_acc_of_{}'.format(c_name), c_val_acc)

            model.train(True)  # train mode

            val_accuracy = float(torch.mean(torch.FloatTensor(total_val_accuracies)))

            del x_val, y_val, pred_val, pred_vals, y_vals
            del val_dataloader_iter
            print("%d th iter accuracy: %.3f" % (i, val_accuracy))
            # print('------------------------save model------------------------')
            if val_accuracy > best_accuracy:
                # save best model
                best_accuracy = val_accuracy
                best_accuracies_each_c = val_accuracies_each_c
                best_mean_val_accuracies = mean_val_accuracies
                best_total_val_accuracies = total_val_accuracies
                options = io_utils.get_model_options_from_args(args, i)
                # dict to save
                model_dict = {'model': model.cpu().state_dict()}
                optimizer_dict = {'optimizer': optimizer.state_dict()}

                # save best checkpoint
                io_utils.save_checkpoints(args.save_dir, options, i, model_dict, optimizer_dict,
                                          logger, best=True)
                # ship to cuda
                model = model.cuda(args.gpu)

                # save best result into textfile
                contents = [' '.join(sys.argv) + '\n',
                            "best accuracy: {:.2f}%\n".format(best_accuracy)]
                best_accuracy_each_c = best_accuracies_each_c[0]
                best_mean_val_accuracy = best_mean_val_accuracies[0]
                best_total_val_accuracy = best_total_val_accuracies[0]
                contents.extend(["{}\n".format(args.finetune_dataset),
                                 "best total acc: {:.2f}%\n".format(100 * best_total_val_accuracy),
                                 "best mean acc: {:.2f}%\n".format(100 * best_mean_val_accuracy),
                                 'Best Accs: ' + ''.join(["{:.2f}% ".format(100 * c_val_acc)
                                                          for _, c_val_acc in best_accuracy_each_c]) + '\n'])

                best_result_path = os.path.join('./output', '{}_best_result.txt'.format(
                    os.path.splitext(os.path.basename(__file__))[0]))
                with open(best_result_path, 'a+') as f:
                    f.writelines(contents)

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
    for cls in val_accuracy_each_c:
        print(cls)
    print(total_val_accuracy)

if __name__ == '__main__':
    main()
