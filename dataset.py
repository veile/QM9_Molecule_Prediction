import os
import numpy as np
import pandas as pd
import math

from ase.io.cube import read_cube
from torch.utils.data.dataset import Dataset

class MolecularDataset(Dataset):
    def __init__(self, data_dir):
        self.systems = pd.DataFrame()
        self.data_dir = data_dir
        
        print("Loading data...")
        # Number of files
        self.names = os.listdir(data_dir)
        self.names.sort()
        self.names = np.array(self.names)
        N = len(self.names)
        
        pickle_name = "molecule_set_%i_entries.pkl" %(N-1)
        mask = np.isin(self.names, pickle_name)

        if mask.sum() == 1:
          self.names = self.names[ self.names != pickle_name]
          N = len(self.names)
          print("Pickle file exist - Loading pickle file with %i entries" %(N))
          self.systems = pd.read_pickle(data_dir+pickle_name)

        else:
          i=0
          for file in self.names:
              with open(data_dir+file, 'r') as f:
                  s = read_cube(f)
                  self.systems = self.systems.append(s, ignore_index=True)
                  
                  i+=1
                  print("Loaded %i of %i" %(i, N))
          self.systems = self.systems.drop(columns="origin")
          pickle_name = "molecule_set_%i_entries.pkl" %N
          print("Saving dataset to \"%s\" for future work" %(data_dir+pickle_name) )
          self.systems.to_pickle(data_dir+pickle_name)        
                
    def __getitem__(self, index):
        return self.systems.loc[index]
                
    def __len__(self):
        return len(self.systems) 

    def no_electrons(self):
        electrons = self.systems['data'].apply(lambda n: 0.05**3*np.sum(n))
        return electrons.values

    def get_distance(self):
        def distance(a):
            pos = a.get_positions()
            dist = 0
            for i in range(3):
                d = np.max( pos[:,i] ) - np.min( pos[:, i]) 
                if d > dist:
                    dist = d
            return dist
        
        distances = self.systems['atoms'].apply(distance)
        return distances.values

    def clean(self):
        # Checking if shape is equal for all entries
        if np.size( np.unique( self.systems['data'].apply(np.shape).values ) ) == 1:
            print("Dataset already cleaned!")
        
        else:
            self.rm_entries()
            self.pad_data()
            
            self.systems['data'] = self.systems['data'].apply(lambda x: x.astype(np.float16))
            
            N = len(self)
            pickle_name = "molecule_set_%i_entries.pkl" %N
            print("Saving dataset to \"%s\" for future work" %(self.data_dir+pickle_name) )
            self.systems.to_pickle(self.data_dir+pickle_name)


    def rm_entries(self, max_size=6):
        N = len(self)
        dist = self.get_distance()
        idx = dist < max_size
        
        self.systems = self.systems[idx]
        self.names = self.names[idx]
        
        print("Removing %i entries" %(N-idx.sum()) )

    def pad_data(self):
        dr = 0.05
        dist = self.get_distance()
        mx = math.ceil( (dist.max()+4)/dr )

        def pad(n, mx):
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
        
        self.systems['data'] = self.systems['data'].apply(lambda n: pad(n, mx) )
