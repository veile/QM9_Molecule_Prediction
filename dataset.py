import os
import numpy as np
import math

from ase.io.cube import read_cube
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class MolecularDataset(Dataset):
    def __init__(self, data_dir):    
    
        self.data_dir = data_dir
        self.precision = np.float32
        
        print("Dataset initiated")
        # Number of files
        self.names = os.listdir(data_dir)
        self.names.sort()
        self.names = np.array(self.names)   
        
        
#        self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need. 

                
    def __getitem__(self, index):
        file = self.names[index]
        with open(self.data_dir+file, 'r') as f:
            self.a, n, _ = read_cube(f).values() # Only takes atoms and electron density
        n, self.flag = self._clean(self.a, n, max_size=6)
        
        return n
#        return [a, n, flag]
                
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
        
    def ground_truth(self, final_grid = 163, radius=8, gridsize=200):
        X, Y, Z = np.ogrid[:gridsize, :gridsize, :gridsize]
        
        true = np.zeros((6,200,200,200))
        atoms_pos = np.round(self.a.get_positions()*20).astype(int)+100
        atomic_numbers = self.a.get_atomic_numbers()
        # HCONF
        atomic_dict = {
                1 : 1,
                6 : 2,
                8 : 3,
                7 : 4,
                9 : 5}
    
        for i in range(len(self.a)):
            dist_from_center = np.sqrt((X - atoms_pos[i,0])**2 + (Y-atoms_pos[i,1])**2 + (Z-atoms_pos[i,2])**2)
    
            mask = dist_from_center <= radius
            true[atomic_dict[atomic_numbers[i]]] = true[atomic_dict[atomic_numbers[i]]] + mask
        
        # Cropping
        mid = gridsize/2
        ds = final_grid/2
        
        true = true[:, (mid-math.floor(ds)):(mid+math.ceil(ds))
                     , (mid-math.floor(ds)):(mid+math.ceil(ds))
                     , (mid-math.floor(ds)):(mid+math.ceil(ds)) ]
                      
        return true