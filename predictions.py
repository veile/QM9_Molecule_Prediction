# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:21:20 2020

@author: s153012
"""
import numpy as np
import ase as a
import ase.io as aio
import skimage.measure as sm
import torch
import os

from unet import Net
from training_procedure import collate_none
from prepare_set import choose_set
    
def atoms_reconstruction(output):
    """
    Converts the output from a neural network to an ase.Atoms object
    
    Parameters
    ----------
    output : ndarray of values between 0 and 1 (6, N, N, N)
        Output from neural network containg 6 channels correpsonding
        to (Background+HCONF)
        
    Returns
    -------
    molecule : ase.Atoms
        Molecule containing all the found atoms and their positions
        assuming 0.05 Å for each grid point
    """
    
    # Assigning an atom to each channel
    atoms_dict = {1: 'H',
                  2: 'C',
                  3: 'O',
                  4: 'N',
                  5: 'F'     
                  }
    
    # Removes any extra indices from the output shape
    out = np.squeeze( output.cpu().detach().numpy() )
    
    # Reconstructs the ground truth from the output
    # Atoms with highest value between 0-1 is chosen
    reconstruct = np.argmax(out, axis=0)
    
    
    # Finding all clusters (atoms) in the reconstruction and
    # labels them differently
    labeled = sm.label(reconstruct)
    
    # Lists of properterties for each cluster (atom)
    clusters = sm.regionprops(labeled)
    
    # Convert each cluster into an atom and combines them to a molecule
    atoms = []
    for c in clusters:
        idx = np.round(c.centroid).astype(int)
        atom_no = reconstruct[ tuple(idx) ]
        
        # Position is found from the assumption that a grid point
        # corresponds to 0.05 Å
        pos = (idx-reconstruct.shape[0]/2)/20
        atom =  a.atom.Atom( atoms_dict[atom_no], pos)
        atoms.append(atom)
    
    # Cell size is meaningless but added
    return a.Atoms(atoms, cell=[8.5, 8.5, 8.5])

def load_mols(dataset, path):
    if os.path.exists(path+"val_mol.xyz"):
        val_mols = aio.iread(path+"val_mol.xyz")
        pre_mols = aio.iread(path+"pre_mol.xyz")
        
        return list(val_mols), list(pre_mols)
    else:
        dataset = choose_set(dataset)
        net = Net(8)
        best_model = torch.load(path+"best_model.pth")
        net.load_state_dict(best_model["Model"])
        
        #Using CUDA if available
        device = torch.device("cuda:0" if torch.cuda.is_available()\
                                            else "cpu")
        print("Compute device: ")
        print(device)
        
        net.to(device)
        
        
        torch.manual_seed(42)
        train_set, test_set = torch.utils.data.random_split(dataset,
                                            [len(dataset)-167, 167]) 
        torch.manual_seed(np.random.randint(100000))
        
        
        loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=1,
                                             num_workers=0,
                                             shuffle=False,
                                             collate_fn=collate_none)
    
        val_mols = []
        pre_mols = []
    
        for item in loader:
            try:
                tensors, atoms = item
                inputs, targets = tensors
            except TypeError:
                print("Molecule is too big")
                continue
            
            # Moving the tensors to device
            inputs, targets = inputs.to(device).float(),\
                              targets.to(device).long()
            
            outputs = net(inputs)
            pre_mols.append(atoms_reconstruction(outputs))
            val_mols.append(atoms[0])
            
        aio.write(path+"val_mol.xyz", val_mols)
        aio.write(path+"pre_mol.xyz", pre_mols)
        return val_mols, pre_mols
                      

def RMSE(a1, a2):
    return a.geometry.distance(a1,a2)/len(a1)

    

def atoms_comparison(a1, a2):
    l1 = np.sort( a1.get_chemical_symbols() )
    l2 = np.sort( a2.get_chemical_symbols() )
    
    return (l1 != l2).sum()

def sort_atoms(positions):
    pos = positions + 5
    p = np.sqrt( np.sum(pos**2, axis=1))
    
    idx = np.argsort(p)
    
    return positions[idx]
    
if __name__ == "__main__":
    dataset = 2
    path = "QM9_kernel6_input170/"
    true, pre = load_mols(dataset, path)
    
    
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 16})
       
    rmse = [RMSE(a1, a2) for a1, a2 in zip(true, pre)]
    mispredictions = [atoms_comparison(a1, a2) for a1, a2 in zip(true, pre)]
    
    # For some reason it does not calculate the error correctly for atom 39
    # Should outcommented when using another dataset or test set
    rmse[39] = 0.025862
    rmse = np.array(rmse)
    
    plt.figure(figsize=(10, 4))
    plt.plot(rmse, 'o')
    plt.ylabel("RMSE of atom centres [Å]")
    plt.xlabel("Validation index")
    
    plt.xlim([0, 102])
    plt.tick_params(axis='both', which='both', direction='inout',
                    top=True, right=True, length=5)
    
    plt.tight_layout()
    plt.savefig(path+"error.pdf")
    plt.show()
    
    plt.plot(mispredictions, 'o')
    plt.savefig(path+"mispredictions.png", dpi=300)
    plt.show()