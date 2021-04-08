# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import numpy as np
import os
from os.path import join
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


# give data_path as office_home_root/domain
class DataLoader(data.Dataset):
    def __init__(self, data_path, split='train', classes=1000, ssl=True):
        self.data_path = data_path
        self.split = split
        self.names, _ = self.get_dataset_list(data_path)
        self.N = len(self.names)
        self.ssl = ssl
        if(self.ssl):
            self.permutations = self.__retrive_permutations(classes)
        else:
            self.classes = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
                   'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
                   'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
                   'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
                   'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                   'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                   'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = join(self.data_path, self.names[index])
        img = Image.open(framename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        if (self.ssl):
            order = np.random.randint(len(self.permutations))
            data = [tiles[self.permutations[order][t]] for t in range(9)]
        else:
            data = [tiles[t] for t in range(9)]

        data = torch.stack(data, 0)
        if (self.ssl):
            return data, int(order), tiles
        else:
            label = self.class_to_idx[framename.split('/')[5]]
            return data, label


    def __len__(self):
        return len(self.names)

    def get_dataset_list(self, path):
        label_list = os.listdir(path)
        file_names = []
        labels = []
        for label in label_list:
            f_list = os.listdir(os.path.join(path, label))
            f_num = np.round(0.8 * len(f_list)).astype(np.uint8)
            if (self.split == 'train'):
                f_list = f_list[:f_num]
            else:
                f_list = f_list[f_num:]
            for f in f_list:
                if (f.endswith('.jpg')):
                    file_names.append(join(label, f))
                    labels.append(label)
        return file_names, labels

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


if __name__ == '__main__':
    dataset = DataLoader('/data/jihun/OfficeHomeDataset_10072016/Clipart', classes=100)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, item in enumerate(dataloader):
        print(item[0].shape)
        print(item[1])
        print(dataset.permutations[item[1][0]])
        break
