from dataset import MolecularDataset, default_collate
from unet import Net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading trained model
PATH = './best_model.pth'

net = Net(8)
net.load_state_dict(torch.load(PATH)['Model'])
net.to(device)

inputs, targets = dataset[17]

inputs = torch.from_numpy(inputs[np.newaxis, :, :, :, :])
targets = torch.from_numpy(targets[np.newaxis, :, :, :])
inputs, targets = inputs.to(device).float(), targets.to(device).long()

outputs = net(inputs)

ground = targets.detach().cpu().numpy().reshape(dataset.output_grid, dataset.output_grid, dataset.output_grid)
n = inputs.detach().cpu().numpy().reshape(dataset.input_grid, dataset.input_grid, dataset.input_grid)

# Plotting the output
layer = 95
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

name="3h_run_no19"
plt.savefig("Figures/%s.png" %name, dpi=300)
plt.show()

# Saving the arrays for plotting 3d
np.savez("Results/%s.npz" %name, gt=ground, out=output)
