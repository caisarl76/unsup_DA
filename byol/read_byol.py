import torch
import torchvision
import pickle
from collections import OrderedDict

from model.rot_resnetdsbn import get_rot_model

weight_path = 'byol_r50_bs256_accmulate16_ep300-5df46722.pth'






if __name__ == '__main__':
    weight_path = 'byol_r50_bs256_accmulate16_ep300-5df46722.pth'
    model = get_rot_model('resnet50dsbn', num_domains=4)
    model.eval()
    pre_weight = torch.load(weight_path)['state_dict']
    missig_key = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    bn_list = ['bns.0', 'bns.1', 'bns.2', 'bns.3']
    dsbn_dict = []
    model_dict = model.named_parameters()

    new_dict = OrderedDict()

    for name in pre_weight:
        split = name.split('.')

        if ('bn' in name):

            for bn in bn_list:
                new_split = name.split('.')
                new_split.insert(-1, bn)
                new_name = '.'.join(new_split)
                print(new_name)
                new_dict[new_name] = pre_weight[name]
        elif ('downsample.1' in name):
            for bn in bn_list:
                new_split = name.split('.')
                new_split.insert(-1, bn)
                new_name = '.'.join(new_split)
                print(new_name)
                new_dict[new_name] = pre_weight[name]
        #         print(name)
        else:
            new_dict[name] = pre_weight[name]

    model.load_state_dict(new_dict, strict=False)
    # model.load_state_dict(dict([(n, p) for n, p in new_dict.items()]), strict=False)
