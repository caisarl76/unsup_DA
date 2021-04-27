import torch.utils.data as data
from PIL import Image
import os
import os.path
import random

import torchvision.transforms.functional as TF
import errno
import numpy as np
from torchvision import transforms
import torch
import codecs
from utils.io_utils import download_url, check_integrity
import zipfile
import h5py
import functools

domain_dict = {'RealWorld': 1, 'Clipart': 0}
OFFICEHOME_DIR = '/data/jihun/OfficeHomeDataset_10072016'


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def rotate(image, i):
    rot_img = TF.rotate(image, i * 90)
    return rot_img


def get_transform(split):
    if split == 'train':
        transform = transforms.Compose([

            # transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),

            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


class rot_dataset(data.Dataset):
    def __init__(self, root, num_domains, domain, split):
        self.extensions = ['jpg', 'jpeg', 'png']
        self.transform = get_transform(split)
        domain_root_dir = []
        for i in range(num_domains):
            domain_root_dir.append(os.path.join(root, domain[i]))

        self.classes = [0, 1, 2, 3]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print('make dataset')
        self.samples = make_dataset(domain_root_dir, self.transform, self.class_to_idx, self.extensions)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + domain_root_dir + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

        self.root = domain_root_dir
        self.loader = Image.open
        for i in range(num_domains):
            self.domain = domain[i]
        print('load dataset complete')

    def __getitem__(self, index):
        path, angle = self.samples[index]
        img = self.loader(path).convert('RGB')
        img = img.rotate(90 * angle)
        img = self.transform(img)
        return img, angle

    def __len__(self):
        return len(self.samples)


def make_dataset(dirs, transform, class_to_idx, extensions, include_dir=False):
    items = []
    for dir in dirs:
        print(dir)
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        angles = np.random.permutation(4)
                        for i in angles:
                            path = os.path.join(root, fname)
                            item = (path, i)
                            items.append(item)
    return items


if __name__ == '__main__':
    dataset = rot_dataset(OFFICEHOME_DIR, 2, ['RealWorld', 'Clipart'], 'train')
    print(len(dataset))
    dataset = rot_dataset(OFFICEHOME_DIR, 3, ['RealWorld', 'Clipart', 'Art'], 'train')
    print(len(dataset))