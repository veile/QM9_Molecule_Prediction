# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:42:09 2020

@author: Thomas
"""

import matplotlib.pyplot as plt
import torch
from dataset import MolecularDataset, collate_none
from unet import Net

# Loading trained model
PATH="QM9_net_100_sigmoid"

net = Net(8)
net.load_state_dict(torch.load(PATH+".pth",  map_location=torch.device('cpu')))


# Loading dataset
data_dir = "Data/"
dataset =  MolecularDataset(data_dir)

n, ground =  dataset[5]

plt.figure( figsize=(12,12))

plt.subplot(431)
plt.title("Ground truth")
plt.imshow( ground[80, :, :], origin='lower')
plt.colorbar()

plt.subplot(432)
plt.title("Electron Density")
plt.imshow( n[0, 104, :, :], origin='lower')
plt.colorbar()


output = net( torch.tensor( n.reshape(1,1,200,200,200) ).float() )
output = output.detach().numpy()
for c in range( output.shape[1] ):
    plc = 433+c
    plt.subplot(plc)
    plt.title("Channel %i" %(c+1))
    plt.imshow( output[0, c, 80, :, :], origin='lower')
    plt.colorbar()

plt.tight_layout()
plt.savefig("comparison_%s.png" %PATH, dpi=300)
plt.show()
