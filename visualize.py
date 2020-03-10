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
PATH="./QM9_net.pth"

net = Net(8)
net.load_state_dict(torch.load(PATH,  map_location=torch.device('cpu')))


# Loading dataset
data_dir = "Data/"
dataset =  MolecularDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)


n, ground =  next(iter(loader))

plt.figure( figsize=(12,12))

plt.subplot(431)
plt.title("Ground truth")
plt.imshow( ground[0, 80, :, :], origin='lower')


output = net(n.float())
output = output.detach().numpy()
for c in range( output.shape[1] ):
    plc = 432+c
    plt.subplot(plc)
    plt.title("Channel %c" %(c+1))
    plt.imshow( output[0, c, 80, :, :], origin='lower')

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.show()