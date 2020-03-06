from dataset import MolecularDataset, collate_none
from unet import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


data_dir = "Data/"
dataset =  MolecularDataset(data_dir)

loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)

# Training the Neural Network
#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)

net = Net(8)
net.to(device)

# Training the neural network

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.SGD(net.parameters(), lr=0.5)#, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(loader):
        print("-----------------------------------------------")        
        inputs, targets  = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
        
print('Finished Training')
