from dataset import MolecularDataset
from unet import Net
import numpy as np
#from mayavi import mlab
import torch
#import matplotlib.pyplot as plt

data_dir = "Data/"
dataset = MolecularDataset(data_dir)

N_before = np.sum( dataset.no_electrons() )

dataset.clean()

N_after = np.sum( dataset.no_electrons() )

print("Number of electrons before cut: %.5f" %N_before)
print("Number of electrons after cut: %.5f" %N_after)
print("Total removed electrons: %.5f" %(N_before - N_after) )

"""
def electron3d(dataset, n):
    values = dataset[n]['data']
    print("Looking at " + str(dataset[n].atoms.symbols) )
    mlab.contour3d(values, contours=50, transparent=True, vmin=-0.1)#, extent=[0, 198, 0, 198, 0, 198])
    mlab.pipeline.volume(mlab.pipeline.scalar_field(values))

electron3d(dataset, 5)
"""

# Training on the dataset
data = np.stack( dataset.systems['data'].values ).astype(np.float16)
# Checking memory
from sys import getsizeof
print(getsizeof(data)/2**30) #Converting bytes to gigabytes
print(getsizeof(dataset.systems)/2**30)
print(dataset.systems.dtypes)
print(data.shape)

#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)

net = Net(2)
net.to(device)
# Inputting first observation
input = torch.tensor(data[0].reshape(1, 1, 198, 198, 198)).to(device)
print(input.shape)

out = net(input)
print(out.shape)

"""
# Plotting slices of the density
sli = 75
re = int( 156/198*sli )

plt.figure(figsize=(12,6))
plt.subplot(131)
plt.imshow( input.detach().numpy()[0, 0, sli, :, :] )

plt.subplot(132)
plt.imshow( out.detach().numpy()[0, 0, re, :, :] )

plt.subplot(133)
plt.imshow( out.detach().numpy()[0, 1, re, :, :] )
plt.show()
"""
