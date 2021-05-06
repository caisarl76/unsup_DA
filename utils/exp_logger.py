import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.log import Log

__all__ = ['ExpLogger']


class ExpLogger:
    def __init__(self, save_root, exist_ok=False):
        self.save_root = save_root
        self.main_log = Log(os.path.join(save_root, 'log.txt'))
        os.makedirs(self.save_root, exist_ok=exist_ok)

        self.tensor_log = SummaryWriter(self.save_root)

    def save_args(self, args):
        args_log = ''
        for argument in args.__dict__.keys():
            args_log += '--%s %s \\\n' % (argument, args.__dict__[argument])

        args_path = os.path.join(self.save_root, 'args.txt')
        args_logger = Log(args_path)
        args_logger.write(args_log, end='', add_time=False)

    def save_cfg(self, cfg):
        cfg_path = os.path.join(self.save_root, 'config.txt')
        cfg_logger = Log(cfg_path)
        cfg_logger.write(str(cfg), add_time=False)

    def save_src(self, src_root):
        src_save_path = os.path.join(self.save_root, 'src')
        shutil.copytree(src_root, src_save_path)

    def log_model(self, model):
        model_log_path = os.path.join(self.save_root, 'model.txt')
        model_logger = Log(model_log_path)
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        model_logger.write(log=str(model), add_time=False)
        model_logger.write(log='Total number of parameters: %d' % num_params, add_time=False)

    def write(self, log, end='\n', is_print=True, add_time=True):
        self.main_log.write(log, end=end, is_print=is_print, add_time=add_time)

    def save_checkpoint(self, checkpoint, name):
        checkpoint_root = os.path.join(self.save_root, 'weights')
        os.makedirs(checkpoint_root, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_root, name)
        torch.save(checkpoint, checkpoint_path)

        self.main_log.write('Checkpoint is saved to %s' % checkpoint_path)

    def load_checkpoint(self, name):
        checkpoint_path = os.path.join(self.save_root, 'weights', name)
        checkpoint = torch.load(checkpoint_path)

        self.main_log.write('Checkpoint is Loaded from %s' % checkpoint_path)

        return checkpoint