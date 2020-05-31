import numpy as np
import torch
import torch.nn.functional as F

from prepare_set import choose_set
from unet import Net
from training_procedure import collate_none

import os

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})
from matplotlib.colors import LinearSegmentedColormap
colors = [ (0, 0, 0, 0.2), (1, 1, 1), (0, 0, 0), (1, 0, 0), (0, 0, 1)]
atoms_cmap = LinearSegmentedColormap.from_list('atoms', colors, N=5)

#Using CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Compute device: ")
print(device)



def single_check(path, layer, mol_idx):

    name = "results.npz"
    # Compute the output if itis not present
    if os.path.exists(path+name):
        data = np.load(path+name)
        out = np.squeeze( data['out'] )
        gt = np.squeeze( data['gt'] )
    else:
        # Read settings from specs.txt
        with open(path+"specs.txt", 'r') as f:
            content = f.read().splitlines()
            set_idx = int(content[2])
            
        # Sets up neural network
        # Specs.txt does not include mol_idx, which could be an added 
        # feature later
        dataset, _ = choose_set(set_idx, mol_idx)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             num_workers=0, shuffle=False,
                                             collate_fn=collate_none)
        
        # Load neural network
        model = Net(8)
        best_model = torch.load(path+"TrainingProcedure.pth",
                                map_location=device)
        model.load_state_dict(best_model["Model"])
        model.to(device)
        tensors, atoms = next(iter(loader))
        print(atoms)
        inputs, targets = tensors
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        
        outputs = model(inputs)
        
        gt = np.squeeze( targets.detach().cpu().numpy() )
        out = np.squeeze( F.softmax(outputs, 1).detach().cpu().numpy() )
        
        np.savez(path+"results.npz", gt=gt, out=out)
          
    M = gt.shape[0]
    scaled = np.round( M/200*layer ).astype(int)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title("Target")
    plt.imshow(gt[:, :, scaled], cmap=atoms_cmap, vmin=0, vmax=4)
    
    
    plt.subplot(122)
    plt.title("Reconstruction")
    plt.imshow(np.argmax(out, axis=0)[:, :, scaled], cmap=atoms_cmap,
               vmin=0, vmax=4)
    
    plt.colorbar()
        
    plt.tight_layout()
    plt.savefig(path+"results.pdf", dpi=300)
    plt.show()

def validation(PATH):
    N, E = np.loadtxt(PATH+"validation_errors.txt", delimiter=' ').T
    
    i = np.argmin(E)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(N, E, 'o:', ms=4)
    plt.plot(N[i], E[i], 'ro')
    
    plt.grid(which='both')
    
    plt.xlabel("Number of training steps")
    plt.ylabel("Cross Entropy Loss")
    
    plt.xlim([0, N[-1]])
    plt.tick_params(axis='both', which='both', direction='inout',
                    top=True, right=True, length=5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
    
    
    plt.tight_layout()
    plt.savefig(PATH+"validation_errors.pdf")
    plt.show()