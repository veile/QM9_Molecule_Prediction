from dataset import MolecularDataset, collate_none
from unet import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


data_dir = "Data/"
dataset =  MolecularDataset(data_dir)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

# Training the Neural Network
#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)

net = Net(8)
net.to(device)

# Training the neural network

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.5)#, momentum=0.9)
targets_tmp = np.zeros( (157, 157, 157) )

print("----Training begun----")
print("[Epoch, Entry] - Loss")
for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        # Moving the tensors to device
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #print(  (outputs.detach().cpu().numpy().sum(axis=1) == targets.cpu().numpy()).sum() )
        

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
       
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
        
print('Finished Training')
