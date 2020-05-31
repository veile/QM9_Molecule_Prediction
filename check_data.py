from dataset import MolecularDataset
import numpy as np

import tarfile
import io
import ase
import os
from ase.io.cube import read_cube
from ase.visualize import view

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})

tar_filename = "qm9_000xxx_29.cube.tar.gz"

def atom(idx):
    """
    Used to retrieve information about an atom in the molecular subset
    """
    tar = tarfile.open(tar_filename, "r:gz")
    names = tar.getnames()
    names.sort()
    names = np.array(names)  
    
    entry = tar.getmember( names[idx] )
    f = tar.extractfile(entry)
    
    content = f.read()
    cubefile = io.StringIO(content.decode('utf-8'))
    cube = read_cube(cubefile) # Only takes atoms and electron density
    return cube['atoms']


def electron_loss():
    """
    Computes the electron loss as function of input grid
    """
    input_grid = np.linspace(120, 220, 20).astype(int) 
    if not os.path.exists("electronloss.npy"):
        tar = tarfile.open(tar_filename, "r:gz")
        
        names = tar.getnames()
        names.sort()
        names = np.array(names)   
        
        # Number of electrons in the dataset before cropping
        N_sum = 0
        for name in names:
            entry = tar.getmember( name )
            f = tar.extractfile(entry)
            
            content = f.read()
            cubefile = io.StringIO(content.decode('utf-8'))
            cube = read_cube(cubefile)
        
            n = cube['data']*(1./ase.units.Bohr**3)
            print(cube['atoms'])
            print(cube['origin'])
            N_sum += n.sum()*0.05**3
        Ntotal = N_sum
        
        
        # Calculates number of atoms cropping to different input grids
           
        loss = []
        for i in range(20):
            dataset =  MolecularDataset(tar_filename,
                                        input_grid=input_grid[i])
            N_sum = 0
            for i in range(len(dataset)):
                inputs, _, _ = dataset[i]
                N = inputs.sum()*0.05**3
                N_sum += N
            loss.append(Ntotal - N_sum)
            
        np.save("electronloss.npy", loss)
        
    else:
        loss = np.load("electronloss.npy")
    
    # Generating the figure
    plt.figure(figsize=(10,4))
    plt.plot(input_grid, loss, 'o:', ms=4)
    plt.hlines(0, 120, 220, 'k', ls='--', lw=1)
    
    plt.xlim([120, 220])
    plt.tick_params(axis='both', which='both', direction='inout',
                    top=True, right=True, length=5)
    
    plt.xlabel("Input grid size")
    plt.ylabel("Number of electrons lost")
    
#    plt.twinx()
#    plt.semilogy(input_grid, input_grid**3)
#    plt.ylabel("Amount of data points")
    
    plt.tight_layout()
    plt.savefig("electronloss.pdf")
    plt.show()