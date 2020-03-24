from dataset import MolecularDataset, collate_none
from unet import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

start = datetime.now()

tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)

train_set, test_set = torch.utils.data.random_split(dataset, [25, 3]) 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

# Training the Neural Network
#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)

net = Net(8)
net.to(device)

# Training the neural network

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=1e-4)
targets_tmp = np.zeros( (dataset.output_grid, dataset.output_grid, dataset.output_grid) )

print("----Training begun----")
print("[Epoch, Entry] - Loss")
epochs=100
for epoch in range(epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        # Moving the tensors to device
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs.min(), outputs.max() )

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
       
        print('[%d, %5d]      %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
        
    print("Epoch no %i finished" %(epoch+1))
    validation_error = 0.0
    for i, (inputs, targets) in enumerate(test_loader):
        # Moving the tensors to device
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        
        outputs = net(inputs)
        validation_error += criterion(outputs, targets)/len(test_set)
    print("Validation error after epoch %i is: %.3f" %(epoch+1, validation_error ))
        
print('Finished Training')
PATH = './QM9_net.pth'
print("Saving model to %s" %PATH)
torch.save(net.state_dict(), PATH)

print("Elapsed time:")
elapsed = datetime.now() - start
print(elapsed)
