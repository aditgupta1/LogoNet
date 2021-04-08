"""
Original code from github: malzantot/Pytorch-conditional-GANs
"""

import os
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms

# from google.colab import drive
# drive.mount('/content/drive')
# data_path = '/content/drive/MyDrive/College/cs452_prj'
data_path = './'    # uncomment line 20-22 and comment this for google colab runs

batch_size = 32
lr = 0.01
epochs = 10
nz = 100
nc = 3  # number of channels
save_every = 1
print_every = 50
save_dir = os.path.join(data_path, 'results')
samples_dir = os.path.join(data_path, 'samples')

# tells PyTorch to use an NVIDIA GPU, if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(samples_dir):
    os.mkdir(samples_dir)

INPUT_SIZE = 28 * 28
SAMPLE_SIZE = 80
NUM_LABELS = 13

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(nc, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1  = nn.Linear(64*28*28+1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(NUM_LABELS, 1000)

    def forward(self, x, labels):
        # batch_size = x.size(0)
        # x = x.view(batch_size, nc, 28,28)
        x = self.conv1(x)   # [bs, 32, 28, 28]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)   # [bs, 64, 28, 28]
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64*28*28)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)

class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(NUM_LABELS, 1000)
        self.fc = nn.Linear(self.z_dim+1000, 64*28*28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, nc, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x
        

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_path, 'images'),
    transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
    )
train_loader = DataLoader(train_dataset, shuffle=True,
    batch_size=batch_size)

model_d = ModelD().to(device)
model_g = ModelG(nz).to(device)
criterion = nn.BCELoss()
input = torch.FloatTensor(batch_size, INPUT_SIZE).to(device)
noise = torch.FloatTensor(batch_size, (nz)).to(device)

fixed_noise = torch.FloatTensor(SAMPLE_SIZE, nz).normal_(0,1).to(device)
fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS).to(device)
for i in range(NUM_LABELS):
    for j in range(SAMPLE_SIZE // NUM_LABELS):
        fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

label = torch.FloatTensor(batch_size).to(device)
one_hot_labels = torch.FloatTensor(batch_size, 10).to(device)

optim_d = optim.SGD(model_d.parameters(), lr=lr)
optim_g = optim.SGD(model_g.parameters(), lr=lr)
fixed_noise = Variable(fixed_noise)
fixed_labels = Variable(fixed_labels)

real_label = 1
fake_label = 0

for epoch_idx in range(epochs):
    model_d.train()
    model_g.train()
        

    d_loss = 0.0
    g_loss = 0.0
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        batch_size = train_x.size(0)
        # train_x = train_x.view(-1, INPUT_SIZE)
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        input.resize_as_(train_x).copy_(train_x)
        label.resize_(batch_size).fill_(real_label)
        one_hot_labels.resize_(batch_size, NUM_LABELS).zero_()
        one_hot_labels.scatter_(1, train_y.view(batch_size,1), 1)
        inputv = Variable(input)
        labelv = Variable(label)

        output = model_d(inputv, Variable(one_hot_labels))
        optim_d.zero_grad()
        errD_real = criterion(output, labelv)
        errD_real.backward()
        realD_mean = output.data.cpu().mean()
        
        one_hot_labels.zero_()
        rand_y = torch.from_numpy(np.random.randint(0, NUM_LABELS, size=(batch_size,1))).to(device)
        one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
        noise.resize_(batch_size, nz).normal_(0,1)
        label.resize_(batch_size).fill_(fake_label)
        noisev = Variable(noise)
        labelv = Variable(label)
        onehotv = Variable(one_hot_labels)
        g_out = model_g(noisev, onehotv)
        output = model_d(g_out, onehotv)
        errD_fake = criterion(output, labelv)
        fakeD_mean = output.data.cpu().mean()
        errD = errD_real + errD_fake
        errD_fake.backward()
        optim_d.step()

        # train the G
        noise.normal_(0,1)
        one_hot_labels.zero_()
        rand_y = torch.from_numpy(
            np.random.randint(0, NUM_LABELS, size=(batch_size,1))).to(device)
        one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1).to(device)
        label.resize_(batch_size).fill_(real_label)
        onehotv = Variable(one_hot_labels)
        noisev = Variable(noise)
        labelv = Variable(label)
        g_out = model_g(noisev, onehotv)
        output = model_d(g_out, onehotv)
        errG = criterion(output, labelv)
        optim_g.zero_grad()
        errG.backward()
        optim_g.step()
        
        d_loss += errD.item()
        g_loss += errG.item()
        if batch_idx % print_every == 0:
            print(
            "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                    realD_mean))

            g_out = model_g(fixed_noise, fixed_labels).data.view(
                SAMPLE_SIZE, nc, 28,28).cpu()
            save_image(g_out,
                '{}/{}_{}.png'.format(
                    samples_dir, epoch_idx, batch_idx))


    print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
        d_loss, g_loss))
    if epoch_idx % save_every == 0:
        torch.save({'state_dict': model_d.state_dict()},
                    '{}/model_d_epoch_{}.pth'.format(
                        save_dir, epoch_idx))
        torch.save({'state_dict': model_g.state_dict()},
                    '{}/model_g_epoch_{}.pth'.format(
                        save_dir, epoch_idx))
