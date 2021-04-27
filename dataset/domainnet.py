from os import path
import os
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torch

def rotate(image, i):
    rot_img = TF.rotate(image, i * 90)
    return rot_img

def read_domainnet_data(dataset_path, domain_name, split="train", ssl=False):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            if ssl:
                # pass
                for i in np.random.permutation(4):
                    data_paths.append(data_path)
                    data_labels.append(i)
            else:
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)

    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name, ssl=False, domain_root=None):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.ssl = ssl
        if (self.ssl):
            # pass
            self.classes = [0, 1, 2, 3]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        else:
            self.classes = sorted(os.listdir(domain_root))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        label = self.data_labels[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.ssl:
            img = img.rotate(90 * label)
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(dataset_path, domain_name, ssl=False):
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train", ssl=ssl)
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test", ssl=ssl)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    domain_root = path.join(dataset_path, domain_name)
    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name, ssl=ssl,
                              domain_root=domain_root)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name, ssl=ssl,
                             domain_root=domain_root)
    return train_dataset, test_dataset

    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            shuffle=True)
    # test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                           shuffle=True)
    # return train_dloader, test_dloader
