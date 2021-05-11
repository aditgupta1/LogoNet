from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt

TOTAL_LOGO_COUNT = 60000
BATCH_SIZE = 32
NUM_EPOCHS = 10
nc = 3 #number of channels in input image
nz = 100 #size of z latent space
ngf = 64 # Size of feature maps in generator
ndf = 64 #Size of feature maps in discriminator
DATA_PATH = 'LLD-icon-sharp.hdf5'

#uncomment these three lines for colab (after adding LLD-icon-sharp.hdf5 to your drive)
# from google.colab import drive
# drive.mount('/content/gdrive')
# DATA_PATH = '/content/gdrive/My Drive/LLD-icon-sharp.hdf5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#input is path to data and number of logos for train set 
def load_hdf5_data(path, logo_count=None):
    #load images from hdf5 file format 
    hdf5_file = h5py.File(path, 'r')
    images, labels = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
    
    #if no logo_count given, return all logos
    if logo_count == None:
        logo_count = len(labels)
        
    images, labels = images[:logo_count], labels[:logo_count]
    #images = images.reshape(logo_count, -1) #reshape elements to be logo_count x 3072 shape
    images = images/255
    dataset = list(map(lambda x, y:[x,y], images, labels))
    return images, labels, dataset

#prints image whos original input was 3 x 32 x 32 that is now 1 x 3072 using plt
def print_transposed_image(image):
    plt.imshow(np.transpose(image.reshape(3,32,32), (1, 2, 0)))
    plt.show()
    return

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( 64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),

            # nn.ConvTranspose2d( 64 * 8, 64, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),

            nn.ConvTranspose2d(64*8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        print("bf1",x.shape)
        x = self.main(x)
        x = torch.squeeze(x)
        print("af", x.shape)
        print("af2", torch.squeeze(x).shape)
        return x
images, labels, dataset = load_hdf5_data(DATA_PATH,TOTAL_LOGO_COUNT)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

generator = Generator().to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optimizerG = optim.Adam(generator.parameters(), lr=0.0002)
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

for epoch in range(NUM_EPOCHS):
    G_loss = []
    D_loss = []
    for i, (data, labels) in enumerate(dataloader):
        data,labels = data.float().to(device), labels.float().to(device)
        print(data.shape)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        discriminator.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1,
                           dtype=real_cpu.dtype, device=device)

        output = discriminator(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(0)
        output = discriminator(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        output = discriminator(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, NUM_EPOCHS, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # if i % 100 == 0:
        #     vutils.save_image(real_cpu,
        #             'real_samples.png' ,
        #             normalize=True)
        #     fake = generator(fixed_noise)
        #     vutils.save_image(fake.detach(),
        #             'fake_samples_epoch_%03d.png' % (epoch),
        #             normalize=True)


