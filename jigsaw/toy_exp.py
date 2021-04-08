from os.path import join
from torchvision import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable

from network.AlexnetDSBN import AlexnetDSBN
from network.network import Network
from Dataset.data_loader import DataLoader
from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy

model = AlexnetDSBN(classes=65, in_features=0, num_domains=2)
model.cuda()
# print(model)
# model2 = Network(classes=65)


save_root = '~/jhkim/results/dsbn_ori/jigsaw_result/'

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trainpath = join('/data/jihun/OfficeHomeDataset_10072016/', 'RealWorld')
train_data = DataLoader(trainpath, split='train', classes=65, ssl=False)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=40, shuffle=True,
                                           num_workers=5)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

batch_time, net_time = [], []

for epoch in range(10):
    lr = adjust_learning_rate(optimizer, epoch, init_lr=0.001, step=20, decay=0.1)
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images, 0 * torch.ones(images.shape[0], dtype=torch.long).cuda())
        prec1 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1,))
        acc = prec1[0]
        # acc = prec1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = float(loss.cpu().data.numpy())
    print('loss: %1.3f'%(loss))
