#Module that implements vanilla GAN architecture
#dataset used is LLD (Large Logo Dataset)

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim as optim

#hyperparameters
DATA_PATH = '../datasets/LLD-icon-sharp.hdf5'
TOTAL_LOGO_COUNT = 60000
BATCH_SIZE = 32
NUM_EPOCHS = 100

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
    images = images.reshape(logo_count, -1) #reshape elements to be logo_count x 3072 shape
    images = images/255
    dataset = list(map(lambda x, y:[x,y], images, labels))
    return images, labels, dataset

#prints image whos original input was 3 x 32 x 32 that is now 1 x 3072 using plt
def print_transposed_image(image):
    plt.imshow(np.transpose(image.reshape(3,32,32), (1, 2, 0)))
    plt.show()
    return
    
#GAN generator class 
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator,self).__init__()
    
        self.main = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )
      
    def forward(self, x, labels=None):
        x = self.main(x)
        return x 

#GAN discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator,self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128,output_size),
            nn.Sigmoid()
        )
    def forward(self, x, labels=None):
        x = self.main(x)
        return x 

#single step of training for generator 
def train_generator(batch_size):
    generator_optimizer.zero_grad()
    criterion = nn.BCELoss()
    
    noise = torch.randn(batch_size, 100).to(device)
    fake_imgs = generator(noise)
    prediction = discriminator(fake_imgs)
    
    loss = criterion(prediction, torch.ones(batch_size,1).to(device)) #goal of generator is to make discriminator output as many 1's for fake imgs
    loss.backward()
    generator_optimizer.step()
    return loss
    
#single step of training for discriminator 
def train_discriminator(real_imgs, real_labels, batch_size):
    discriminator_optimizer.zero_grad()
    criterion = nn.BCELoss()
    
    noise = torch.randn(batch_size, 100).to(device)
    fake_imgs = generator(noise)
    fake_prediction = discriminator(fake_imgs)
    fake_loss = criterion(fake_prediction, torch.zeros(batch_size,1).to(device)) #goal of discriminator is to output 0 for fake imgs
    fake_loss.backward()
    
    real_prediction = discriminator(real_imgs)
    real_loss = criterion(real_prediction, torch.ones(batch_size,1).to(device)) #other goal of discriminator is to output 1 for real imgs
    real_loss.backward()
    
    discriminator_optimizer.step()
    
    return real_loss + fake_loss


images, labels, dataset = load_hdf5_data(DATA_PATH,TOTAL_LOGO_COUNT)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

generator = Generator(100, 32*32*3)
discriminator = Discriminator(32*32*3, 1)

generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    G_loss = []
    D_loss = []
    for batch_idx, (data,labels) in enumerate(dataloader):
        data,labels = data.float(), labels.float()
        g_loss = train_generator(BATCH_SIZE)
        G_loss.append(g_loss)
        d_loss = train_discriminator(data, labels, BATCH_SIZE)
        D_loss.append(d_loss)
        
        if (batch_idx + 1) % 500 == 0 and (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch}: loss_d: {torch.mean(torch.FloatTensor(D_loss))}, loss_g: {torch.mean(torch.FloatTensor(G_loss))}")
    
    noise = torch.randn(10, 100)
    generated_imgs = generator(noise)
    plt.imshow(np.transpose(generated_imgs[0].reshape(3,32,32), (1, 2, 0)))
    plt.show()
