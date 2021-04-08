"""
Factory method for easily getting dataset by name.
written by wgchang
"""
import utils.custom_transforms as custom_transforms
from torchvision import transforms
from numpy import array

__sets = {}

from dataset.datasets import SVHN, MNIST
from dataset.datasets import OFFICEHOME

# for digit DA
MNIST_DIR = './data/MNIST'
SVHN_DIR = './data/SVHN'
SPLITS = ['train', 'test', 'val']
JITTERS = ['None', ]
for model_name in ['lenet', 'lenetdsbn']:
    for domain in ['mnist', 'svhn']:
        for jitter in JITTERS:
            if domain == 'mnist':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x.repeat([3, 1, 1])),
                ])
                for split in SPLITS:
                    name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                           split=split, jitter=jitter)
                    train = True if split == 'train' else False
                    __sets[name] = (
                        lambda train=train, transform=transform: MNIST(MNIST_DIR, train=train, download=True,
                                                                       transform=transform))

            elif domain == 'svhn':
                transform = transforms.Compose([
                    transforms.Resize([28, 28]),
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ])
                for split in SPLITS:
                    name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                           split=split, jitter=jitter)
                    split = 'test' if split == 'val' else split
                    __sets[name] = (
                        lambda split=split, transform=transform: SVHN(SVHN_DIR, split=split, download=True,
                                                                      transform=transform))


# for digit DA
SPLITS = ['train', 'test', 'val']
JITTERS = ['None', ]

# Office Home datasets
# change "Real World" to "RealWorld" domain_name with space is not allowed!
OFFICEHOME_DIR = '/data/jihun/OfficeHomeDataset_10072016'
JITTERS = ['None', ]
# for model_name in ['alexnet','resnet18dsbn',  'resnet50', 'cresnet50', 'resnet50dsbn', 'cpuanet50', 'cpuanet50dsbn']:
for model_name in ['resnet18dsbn',  'resnet50', 'resnet50dsbn']:
    for split in SPLITS:
        for jitter in JITTERS:
            if model_name == 'alexnet':
                if jitter == 'None':
                    if split == 'train':
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
                else:
                    continue
            else:  # for resnet
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

            for domain in ['Art', 'Clipart', 'Product', 'RealWorld']:
                name = '{model_name}_{domain}_{split}_{jitter}'.format(model_name=model_name, domain=domain,
                                                                       split=split, jitter=jitter)
                __sets[name] = (
                    lambda domain=domain, transform=transform: OFFICEHOME(OFFICEHOME_DIR, domain=domain,
                                                                          transform=transform))


def get_dataset(name):
    if name not in __sets:
        raise KeyError('Unknown Dataset: {}'.format(name))
    # print(name)
    return __sets[name]()
