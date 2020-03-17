from dataset import MolecularDataset, collate_none
from unet import Net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)
loader = torch.utils.data.DataLoader(dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading trained model
PATH = './single_test.pth'

net = Net(8)
net.load_state_dict(torch.load(PATH))
net.to(device)


inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device).float(), targets.to(device).long()

outputs = net(inputs)

ground = targets.detach().cpu().numpy().reshape(dataset.output_grid, dataset.output_grid, dataset.output_grid)
n = inputs.detach().cpu().numpy().reshape(200, 200, 200)

# Plotting the output
layer = 104
layer_scaled = int( dataset.output_grid/dataset.input_grid * layer )

plt.figure( figsize=(12,12))

plt.subplot(431)
plt.title("Ground truth")
plt.imshow( ground[layer_scaled, :, :], origin='lower')
plt.colorbar()

plt.subplot(432)
plt.title("Electron Density")
plt.imshow( n[layer, :, :], origin='lower')
plt.colorbar()

output = F.softmax(outputs, 1)
output = output.detach().cpu().numpy()

for c in range( output.shape[1] ):
    plc = 433+c
    plt.subplot(plc)
    plt.title("Channel %i" %(c+1))
    plt.imshow( output[0, c, layer_scaled, :, :], origin='lower', vmin=0, vmax=1)
    plt.colorbar()

plt.tight_layout()
plt.savefig("single_train_comparison_ADAM_5xcontractive.png", dpi=300)
plt.show()