import os
import random
import time
import timeit
import linecache
import math
from operator import itemgetter
import numpy as np
from numpy import zeros

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as tri
#from mpl_toolkits import mplot3d

#################

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import random
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU available
else:
    device = torch.device("cpu")  # Only CPU available

torch.manual_seed(0)

#########################################################

class CubeDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

##################################
class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.
        
        input shape: (batchsize, x=200, y=96, t=24, c=12)
        output shape: (batchsize, x=200, y=96, t=24, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        #self.fc0 = nn.Linear(12, self.width)
        self.fc0 = nn.Linear(1, self.width)
        """        
        1 ch
        """
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, self.width)
               
        self.fc10 = nn.Linear(self.width, 64*2)
        self.bn10 = nn.BatchNorm1d(64*2)
        self.dropout10 = nn.Dropout(0.3)
        
        self.fc20 = nn.Linear(64*2, 64*2)
        self.bn20 = nn.BatchNorm1d(64*2)
        self.dropout20 = nn.Dropout(0.3)
      
        self.fc30 = nn.Linear(64*2, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        #print("Shape of tensor at the intermediate step:", x.shape)
        
        # Max pool
        x = x.view(x.size(0), -1, x.size(-1)).max(dim=1).values

        # Pass through three fully connected layers with dropout without batch
        x = self.dropout10(F.relu(self.fc10(x)))
        x = self.dropout20(F.relu(self.fc20(x)))
        x = self.fc30(x)
        
        x = torch.sigmoid(x)
        
        return x

#################################
#################################
# Initialize the model, loss function, and optimizer
modes1 = 2 
modes2 = 2 
modes3 = 2 
width = 64

model = SimpleBlock3d(modes1, modes2, modes3, width)
model = model.to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Training loop
num_epochs = 3000
train_losses = []
val_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):

    running_loss = 0.0

    dataiter1 = iter(dataloader_64)
    dataiter2 = iter(dataloader_48)
    dataiter3 = iter(dataloader_56)

    for sub_epoch in range(max(len(dataloader_64), len(dataloader_48), len(dataloader_56))):

        # Training on loader1 data
        try:
            inputs, targets = next(dataiter1)
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f"Epoch: {epoch+1}, Sub-Epoch: {sub_epoch+1}, Dataset: 1, Loss: {loss.item()}")
        except StopIteration:
            pass

        # Training on loader2 data
        try:
            inputs, targets = next(dataiter2)
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f"Epoch: {epoch+1}, Sub-Epoch: {sub_epoch+1}, Dataset: 2, Loss: {loss.item()}")
        except StopIteration:
            pass

	# Training on loader3 data
        try:
            inputs, targets = next(dataiter3)
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f"Epoch: {epoch+1}, Sub-Epoch: {sub_epoch+1}, Dataset: 3, Loss: {loss.item()}")
        except StopIteration:
            pass

    avg_loss = running_loss / (len(dataloader_56) + len(dataloader_48) + len(dataloader_64))
    train_losses.append(avg_loss)

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / (len(dataloader_48) + len(dataloader_56) + len(dataloader_64))}")

# Begin validation for dataloader_40_validation
    model.eval()

    val_loss_64 = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_64_validation:
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            val_loss_64 += loss.item()

# Begin validation for dataloader_48_validation
   
    val_loss_48 = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_48_validation:
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            val_loss_48 += loss.item()

# Begin validation for dataloader_56_validation
    val_loss_56 = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_56_validation:
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            val_loss_56 += loss.item()

    val_losses.append((val_loss_56+val_loss_48+val_loss_64)/(len(dataloader_48) + len(dataloader_56) + len(dataloader_64)))

##################
### save model ###
#torch.save(model.state_dict(), 'model_checkpoint.pth')

# Function for plotting the training loss based on epoch

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend()
    plt.savefig('Loss_History.png', dpi=300)
    plt.clf()
    #plt.show()

plot_loss(train_losses,val_losses)
