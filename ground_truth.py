import numpy as np
from dataset import MolecularDataset
from ase import Atoms
import matplotlib.pyplot as plt

data_dir = "../QM9 data/"
dataset = MolecularDataset(data_dir)
dataset.clean()

###
# create spherical mask 
# Inputs:   atoms - AtomicSystems object for a single molecule
#           radius - desired radius of each atom
#           gridsize - number of points in each grid direction of the cubic grid

def createSphericalMasks( atoms, radius=8, gridsize=200 ):
    
    X, Y, Z = np.ogrid[:gridsize, :gridsize, :gridsize]
    arr = np.zeros((6,200,200,200))
    atoms_pos = np.round(atoms.get_positions()*20).astype(int)+100
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_dict = {
            1 : 1,
            6 : 2,
            8 : 3,
            7 : 4,
            9 : 5}

    for i in range(len(atoms)):
        dist_from_center = np.sqrt((X - atoms_pos[i,0])**2 + (Y-atoms_pos[i,1])**2 + (Z-atoms_pos[i,2])**2)

        mask = dist_from_center <= radius
        arr[atomic_dict[atomic_numbers[i]]] = arr[atomic_dict[atomic_numbers[i]]] + mask
    return arr

a = dataset.systems['atoms']
ground_truth = createSphericalMasks( a[0] )

# %%
test = ground_truth[1,:,:,:]

plt.imshow(test[100,:,:])
