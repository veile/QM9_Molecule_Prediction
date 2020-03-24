# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:42:09 2020

@author: Thomas
"""

import matplotlib.pyplot as plt
import torch
from dataset import MolecularDataset, collate_none
from unet import Net
import torch.nn.functional as F

# Loading trained model
PATH="QM9_net"

net = Net(8)
net.load_state_dict(torch.load(PATH+".pth",  map_location=torch.device('cpu')))


# Loading dataset
tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)

n, ground =  dataset[0]

plt.figure( figsize=(12,12))
layer = 104
layer_scaled = int( dataset.output_grid/dataset.input_grid * layer )

plt.subplot(431)
plt.title("Ground truth")
plt.imshow( ground[layer_scaled, :, :], origin='lower')
plt.colorbar()

plt.subplot(432)
plt.title("Electron Density")
plt.imshow( n[0, layer, :, :], origin='lower')
plt.colorbar()


output = net( torch.tensor( n.reshape(1,1,200,200,200) ).float() )
output = F.softmax(output, 1)
output = output.detach().numpy()
for c in range( output.shape[1] ):
    plc = 433+c
    plt.subplot(plc)
    plt.title("Channel %i" %(c+1))
    plt.imshow( output[0, c, layer_scaled, :, :], origin='lower', vmin=0, vmax=1)
    plt.colorbar()

plt.tight_layout()
plt.savefig("comparison_%s.png" %PATH, dpi=300)
plt.show()
