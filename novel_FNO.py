##### A novel Fourier neural operator framework for classification of multi-sized images #####

#Author: Ali Kashefi (kashefi@stanford.edu)
#Acknowledgements: The first author would like to thank Prof. Gege Wen at Imperial College London for her helpful guidance and discussion on the software engineering aspects of this study.

#Citations:
#If you use the code, please cite the following journal papers:

#@article{kashefi2024novelFNO,
#  title={A novel Fourier neural operator framework for classification of multi-sized images: Application to three dimensional digital porous media},
#  author={Kashefi, Ali and Mukerji, Tapan},
#  journal={Physics of Fluids},
#  volume={36},
#  number={5},
#  year={2024},
#  publisher={AIP Publishing}}

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
	    
#########################################################

def generate_fake_data(cube_size, num_datasets):
    array_list = zeros([num_datasets,cube_size,cube_size,cube_size,1],dtype='f')
    random_numbers = zeros(num_datasets,dtype='f')

    for i in range(num_datasets):
        # Generate a 3D array with random integers 0 or 1
        cube_data = np.random.randint(2, size=(cube_size, cube_size, cube_size))
        array_list[i, :, :, :,0] = cube_data

        # Generate a single random real number between 0 and 1
        random_number = np.random.rand()
        random_numbers[i] = random_number

    return array_list, random_numbers
	
############################################
#In practice, real data must be loaded. However, here we generate fake data solely for testing the model architecture.

data_i = 1250 

# Generate 1250 "fake" datasets of a 40x40x40 cube
input_data_40, output_data_40 = generate_fake_data(40, data_i)

# Generate 1250 "fake" datasets of a 48x48x48 cube
input_data_48, output_data_48 = generate_fake_data(48, data_i)

# Generate 1250 "fake" datasets of a 56x56x56 cube
input_data_56, output_data_56 = generate_fake_data(56, data_i)

# Data spliting for size 40
total_indices = data_i
all_indices = np.random.permutation(total_indices)

training_idx_40 = all_indices[:int(0.8*total_indices)]
validation_idx_40 = all_indices[int(0.8*total_indices):int(0.9*total_indices)]
test_idx_40 = all_indices[int(0.9*total_indices):]

input_training_40, input_test_40, input_validation_40 = input_data_40[training_idx_40,:], input_data_40[test_idx_40,:], input_data_40[validation_idx_40,:]
output_training_40, output_test_40, output_validation_40 = output_data_40[training_idx_40], output_data_40[test_idx_40], output_data_40[validation_idx_40]
                         
#Normalize between [0,1] with sigmoid function 
k_max_40 = np.max(output_training_40)
k_min_40 = np.min(output_training_40)

output_training_40 = (output_data_40 - k_max_40)/(k_max_40 - k_min_40)
output_validation_40 = (output_validation_40 - k_min_40)/(k_max_40 - k_min_40)
output_test_40 = (output_test_40 - k_min_40)/(k_max_40 - k_min_40)

dataset_40 = CubeDataset(input_training_40, output_training_40)
dataloader_40 = DataLoader(dataset_40, batch_size=50, shuffle=True, drop_last=False)

dataset_40_validation = CubeDataset(input_validation_40, output_validation_40)
dataloader_40_validation = DataLoader(dataset_40_validation, batch_size=40, shuffle=True, drop_last=False)

##################### Data spliting for size 48 #####################
total_indices = data_i
all_indices = np.random.permutation(total_indices)

training_idx_48 = all_indices[:int(0.8*total_indices)]
validation_idx_48 = all_indices[int(0.8*total_indices):int(0.9*total_indices)]
test_idx_48 = all_indices[int(0.9*total_indices):]

input_training_48, input_test_48, input_validation_48 = input_data_48[training_idx_48,:], input_data_48[test_idx_48,:], input_data_48[validation_idx_48,:]
output_training_48, output_test_48, output_validation_48 = output_data_48[training_idx_48], output_data_48[test_idx_48], output_data_48[validation_idx_48]
                         
#Normalize between [0,1] with sigmoid function 
k_max_48 = np.max(output_training_48)
k_min_48 = np.min(output_training_48)

output_training_48 = (output_data_40 - k_max_40)/(k_max_48 - k_min_48)
output_validation_48 = (output_validation_48 - k_min_48)/(k_max_48 - k_min_48)
output_test_48 = (output_test_48 - k_min_48)/(k_max_48 - k_min_48)

dataset_48 = CubeDataset(input_training_48, output_training_48)
dataloader_48 = DataLoader(dataset_48, batch_size=50, shuffle=True, drop_last=False)

dataset_48_validation = CubeDataset(input_validation_48, output_validation_48)
dataloader_48_validation = DataLoader(dataset_48_validation, batch_size=40, shuffle=True, drop_last=False)

##################### Data spliting for size 56 #####################
total_indices = data_i
all_indices = np.random.permutation(total_indices)

training_idx_56 = all_indices[:int(0.8*total_indices)]
validation_idx_56 = all_indices[int(0.8*total_indices):int(0.9*total_indices)]
test_idx_56 = all_indices[int(0.9*total_indices):]

input_training_56, input_test_56, input_validation_56 = input_data_56[training_idx_56,:], input_data_56[test_idx_56,:], input_data_56[validation_idx_56,:]
output_training_56, output_test_56, output_validation_56 = output_data_56[training_idx_56], output_data_56[test_idx_56], output_data_56[validation_idx_56]
                         
#Normalize between [0,1] with sigmoid function 
k_max_56 = np.max(output_training_56)
k_min_56 = np.min(output_training_56)

output_training_56 = (output_data_56 - k_max_56)/(k_max_56 - k_min_56)
output_validation_56 = (output_validation_56 - k_min_56)/(k_max_56 - k_min_56)
output_test_56 = (output_test_56 - k_min_56)/(k_max_56 - k_min_56)

dataset_56 = CubeDataset(input_training_56, output_training_56)
dataloader_56 = DataLoader(dataset_56, batch_size=50, shuffle=True, drop_last=False)

dataset_56_validation = CubeDataset(input_validation_56, output_validation_56)
dataloader_56_validation = DataLoader(dataset_56_validation, batch_size=40, shuffle=True, drop_last=False)

#########################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
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
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
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
        
        # Max pool
        x = x.view(x.size(0), -1, x.size(-1)).max(dim=1).values

        # Pass through three fully connected layers with dropout without batch
        x = self.dropout10(F.relu(self.fc10(x)))
        x = self.dropout20(F.relu(self.fc20(x)))
        x = self.fc30(x)
        
        x = torch.sigmoid(x)
        return x

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
num_epochs = 5000
train_losses = []
val_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):

    running_loss = 0.0

    dataiter1 = iter(dataloader_40)
    dataiter2 = iter(dataloader_48)
    dataiter3 = iter(dataloader_56)

    for sub_epoch in range(max(len(dataloader_40), len(dataloader_48), len(dataloader_56))):

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

    avg_loss = running_loss / (len(dataloader_56) + len(dataloader_48) + len(dataloader_40))
    train_losses.append(avg_loss)

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / (len(dataloader_48) + len(dataloader_56) + len(dataloader_40))}")

    model.eval()
	
    # Begin validation for dataloader_40_validation
    val_loss_40 = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_40_validation:
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            val_loss_40 += loss.item()

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

    val_losses.append((val_loss_56+val_loss_48+val_loss_40)/(len(dataloader_48) + len(dataloader_56) + len(dataloader_40)))

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
