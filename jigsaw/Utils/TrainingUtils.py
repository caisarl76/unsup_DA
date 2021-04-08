# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:53:30 2017

@author: bbrattol
"""
import torch




def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    # print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = torch.t(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(torch.mul(correct_k, (100.0 / batch_size)))
    return res

if __name__ == '__main__':
    output = [[0.5, 0.75, 0.5],[0.5, 0.75, 0.5],[0.5, 0.75, 0.5],[0.5, 0.75, 0.5],[0.5, 0.75, 0.5]]
    target = [[1],[0],[1],[1],[0]]
    print(compute_accuracy(torch.tensor(output), torch.tensor(target), topk=(1,3)))