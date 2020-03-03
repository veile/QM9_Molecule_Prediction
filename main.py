from dataset import MolecularDataset, collate_none
from unet import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


data_dir = "Data/"
dataset =  MolecularDataset(data_dir)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                                    collate_fn=collate_none)

entries = 0
for v in loader:
    inputs, targets = v
    print(inputs.shape)
    print(targets.shape)
    entries += inputs.shape[0]
    
print(entries)

# Training the Neural Network
#Using CUDA if available
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Compute device: ")
# print(device)

# net = Net(8)
# net.to(device)

"""
# Training the neural network

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(loader):
        
        # If the molecule is too big, it should not be included in the training
        if not dataset.flag:
            continue
        
        inputs = data.to(device)
        inputs = inputs[np.newaxis, :, :, :, :]
        target = torch.from_numpy( dataset.ground_truth()).to(device)
        target = target[np.newaxis, :, :, :]

        print(target.sum())
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(inputs.float())
        print(output.sum())
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#        if i % 5 == 4:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')
"""
