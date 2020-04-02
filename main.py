from dataset import MolecularDataset, collate_none
from unet import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from os import path
start = time.time()

tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)

test_split = 100
train_split = len(dataset) - test_split

train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split]) 

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

trained_molecules = 0
validation_errors = []
PATH = './QM9_net.pth'
first_loop = ~path.exists(PATH)

if ~first_loop:
    print('Loading previous state...')
    model = torch.load(PATH)
    net.load_state_dict(model['net_state_dict'])
    optimizer.load_state_dict(model['optimizer_state_dict'])
    trained_molecules = model['trained_molecules']
    best_validation_error = model['loss']
    print('Loaded state with %i molecules' % trained_molecules)
    
it = iter(train_loader)
while (time.time() - start) < 23*60*60:  # loop over the dataset multiple times

    running_loss = 0.0
    
    # Recording time for segment
    seg_start = time.time()
    
    inputs, targets = next(it)

    # Moving the tensors to device
    inputs, targets = inputs.to(device).float(), targets.to(device).long()
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
   
    trained_molecules += 1
    print('[%d]      %.3f' % (trained_molecules/len(dataset), running_loss))
    running_loss = 0.0
    
    if (time.time() - seg_start) > 1*60*60:
        validation_error = 0.0
        
        #Testing the model
        for j, (test_inputs, test_targets) in enumerate(test_loader):
            # Moving the tensors to device
            test_inputs, test_targets = test_inputs.to(device).float(), test_targets.to(device).long()
            
            outputs = net(test_inputs)
            validation_error += criterion(outputs, test_targets)/len(test_set)
            print("Validation error after %i molecules is: %.3f" %(trained_molecules, validation_error ))
        validation_errors.append(validation_error)
        
        # if the validation error is lower than current best, replace that model with current model
        if first_loop:
            best_validation_error = validation_error
            first_loop = False
            torch.save({
                        'trained_molecules': trained_molecules,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_validation_error,
                        }, PATH)
            print("Saving model to %s" %PATH)
            print('No. of trained molecules: %i' %trained_molecules)
            print('Validation error: %d' %best_validation_error)
            
        elif validation_error < best_validation_error:      
            best_validation_error = validation_error
            torch.save({
                        'trained_molecules': trained_molecules,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_validation_error,
                        }, PATH)
            print("Saving model to %s" %PATH)
            print('No. of trained molecules %i' %trained_molecules)
            print('Validation error: %d' %best_validation_error)
    seg_start = time.time()
                
print('Training stopped after %d hours' %(time.time()- start)/3600)

