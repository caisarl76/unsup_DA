import dataset.datasets
import os
from os.path import join
from torch.utils import data
import matplotlib.pyplot as plt

from rot_dataset.rot_dataset import rot_dataset
from dataset.datasets import OFFICEHOME_multi
from dataset.domainnet import get_domainnet_dloader

data_pth_dict = {'office-home': 'OfficeHomeDataset_10072016', 'domainnet': 'domainnet'}
domain_dict = {'office-home': {'RealWorld': 0, 'Art': 1, 'Clipart': 2, 'Product': 3},
               'domainnet': {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}}


def get_dataset(dataset='office-home', dataset_root='/data', domain='RealWorld', ssl=False):
    domain_num_dict = domain_dict[dataset]
    data_pth = join(dataset_root, data_pth_dict[dataset])
    if (dataset == 'office-home'):
        if ssl:
            train_dataset = rot_dataset(data_pth, 1, [domain], split='train')
            val_dataset = rot_dataset(data_pth, 1, [domain], split='val')
        else:
            train_dataset = OFFICEHOME_multi(data_pth, 1, [domain], split='train')
            val_dataset = OFFICEHOME_multi(data_pth, 1, [domain], split='val')

    elif (dataset == 'domainnet'):
        if ssl:
            train_dataset, val_dataset = get_domainnet_dloader(data_pth, domain, ssl=True)
        else:
            train_dataset, val_dataset = get_domainnet_dloader(data_pth, domain, ssl=False)

    return train_dataset, val_dataset


# if __name__ == '__main__':
#     # for dom in ['real', 'clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
#     train_dataset, val_dataset = get_dataset(dataset='domainnet', dataset_root='/data/jihun',
#                                              domain='clipart', ssl=True)
#     train_dataloader = data.DataLoader(train_dataset, batch_size=40, shuffle=True,
#                                        num_workers=5, drop_last=True, pin_memory=True)
#     train_dataloader_iter = enumerate(train_dataloader)
#     print(len(train_dataset))
