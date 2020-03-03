import os
import numpy as np
import math

from ase.io.cube import read_cube
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

def collate_none(batch):"
    batch = list( filter (lambda x:x is not None, batch) )
    return default_collate(batch)

class MolecularDataset(Dataset):
    def __init__(self, data_dir, input_grid=200, output_grid=163):    
    
        self.data_dir = data_dir
        self.precision = np.float32
        
        print("Dataset initiated")
        # Number of files
        self.names = os.listdir(data_dir)
        self.names.sort()
        self.names = np.array(self.names)   
        
        # Grid parameters
        self.input_grid = input_grid
        self.output_grid = output_grid

                
    def __getitem__(self, index):
        file = self.names[index]
        with open(self.data_dir+file, 'r') as f:
            a, n, _ = read_cube(f).values() # Only takes atoms and electron density
        n, self.flag = self._clean(a, n, max_size=3)
        target = self._ground_truth(a, radius=8)
        
        if flag:
            # Returning the volumetric data with single channel
            return n[np.newaxis, :, :, :], target[np.newaxis, :, :, :]
        else:
            return None

                
    def __len__(self):
        return len(self.names) 

    def _distance(self, a):
        pos = a.get_positions()
        dist = 0
        for i in range(3):
            d = np.max( pos[:,i] ) - np.min( pos[:, i]) 
            if d > dist:
                dist = d
        return dist

    def _clean(self, a, n, max_size=6):
        # Tells if the entry should be included in training or not
        flag = self._check_entry(a)
        
        if flag:
            n = self._pad_density(n) # Maybe implement loss of electrons
            
        return n, flag


    def _check_entry(self, a, max_size=6):
        if self._distance(a) > max_size:
            return False
        else:
            return True
            
    def _pad_density(self, n, mx=200):
        x, y, z = np.shape(n)

        if x > mx:
            l = (x - mx)/2
            n = n[math.floor(l):(x-math.ceil(l)), :, :]
        if y > mx:
            l = (y - mx)/2
            n = n[:, math.floor(l):(y-math.ceil(l)), :]
        if z > mx:
            l = (z - mx)/2
            n = n[:, :, math.floor(l):(z-math.ceil(l))]

        x, y, z = np.shape(n)

        return np.pad(n, ( (math.floor((mx-x)/2), math.ceil((mx-x)/2)),
                   (math.floor((mx-y)/2), math.ceil((mx-y)/2)),
                   (math.floor((mx-z)/2), math.ceil((mx-z)/2)) ), 'constant', constant_values=0)
        
    def _ground_truth(self, a, final_grid = 163, radius=8, gridsize=200):
        X, Y, Z = np.ogrid[:self.input_grid, :self.input_grid, :self.input_grid]
        
        true = np.zeros((self.input_grid, self.input_grid, self.input_grid))
        
        atoms_pos = np.round(a.get_positions()*20).astype(int)+100
        atomic_numbers = a.get_atomic_numbers()
        
        # HCONF
        atomic_dict = {
                1 : 1,
                6 : 2,
                8 : 3,
                7 : 4,
                9 : 5}
    
        for i in range(len(a)):
            dist_from_center = np.sqrt((X - atoms_pos[i,0])**2 + (Y-atoms_pos[i,1])**2 + (Z-atoms_pos[i,2])**2)
    
            mask = dist_from_center <= radius
            true = true + mask*atomic_dict[atomic_numbers[i] ]
        
        # Cropping
        mid = int(self.input_grid/2)
        ds = self.output_grid/2
        
        true = true[ (mid-math.floor(ds)):(mid+math.ceil(ds))
                    ,(mid-math.floor(ds)):(mid+math.ceil(ds))
                    ,(mid-math.floor(ds)):(mid+math.ceil(ds)) ]
                      
        return true