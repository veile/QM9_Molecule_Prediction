from dataset import MolecularDataset, collate_none
from unet import Net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)
loader = torch.utils.data.DataLoader(dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net(8)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
#optimizer = optim.SGD(net.parameters(), lr=1e-4)

inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device).float(), targets.to(device).long()

epochs=1000
for epoch in range(epochs):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # print statistics
    print(loss.item() )
        
    print("Epoch no %i finished" %(epoch+1))
    
PATH = './single_test.pth'
print("Saving model to %s" %PATH)
torch.save(net.state_dict(), PATH)