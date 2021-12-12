import torch
import torchvision
import torch.nn.functional as functional
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as pyplot
import numpy as np

import cv2

EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


trainset = datasets.FashionMNIST(
    root        = './.data',
    train       = True,
    download    = True,
    transform   = transforms.ToTensor() 
)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 0
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.dimension = 3

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, self.dimension),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.dimension, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0

    for step, (x, label) in enumerate(train_loader):
        noisy_x = add_noise(x)
        noisy_x = noisy_x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)

        encoded, decoded = autoencoder(noisy_x)
        
        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    
    return avg_loss / len(train_loader)


for epoch in range(1, EPOCH +1):
    loss = train(autoencoder, train_loader)
    print ("[Epoch {}] loss : {}". format(epoch, loss))


testset = datasets.FashionMNIST(
    root        = './.data',
    train       = False,
    download    = True,
    transform   = transforms.ToTensor() 
)

sample_data = testset.data[0].view(-1, 28*28)
sample_data = sample_data.type(torch.FloatTensor)/255.0

original_x = sample_data[0]
noisy_x = add_noise(original_x).to(DEVICE)
_, recovered_x = autoencoder(noisy_x)

original_img = np.reshape(original_x.to("cpu").data.numpy(), (28,28))
noisy_img    = np.reshape(noisy_x.to("cpu").data.numpy(), (28,28))
recovered_img = np.reshape(recovered_x.to("cpu").data.numpy(), (28,28))


cv2.imwrite('original_img.png', original_img)

cv2.imwrite('noisy_img.png', noisy_img)

cv2.imwrite('recoverd_img.png', recovered_img)