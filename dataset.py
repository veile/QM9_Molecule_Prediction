import numpy as np
import math
import tarfile
import io

from ase.io.cube import read_cube
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from scipy.ndimage.interpolation import shift

def collate_none(batch):
    batch = list( filter (lambda x:x is not None, batch) )
    return default_collate(batch)


#Output grid 154 for 3x3x3 conv
#Output grid 129 for 4x4x4 conv
#Output grid 108 for 5x5x5 conv
#Output grid 150 for 2 5x5x5 and rest 3x3x3
#Output grid 126 for contractive 5x5x5 and expansive 3x3x3

class MolecularDataset(Dataset):
    def __init__(self, tar_filename, input_grid=200, output_grid=150):    
        
        self.tar = tarfile.open(tar_filename, "r:gz")
        self.precision = np.float32
        
        print("Dataset initiated")
        # Number of files
        self.names = self.tar.getnames()
        self.names.sort()
        self.names = np.array(self.names)   
        
        # Grid parameters
        self.input_grid = input_grid
        self.output_grid = output_grid

                
    def __getitem__(self, index):
        entry = self.tar.getmember( self.names[index] )
        f = self.tar.extractfile(entry)
        if f is not None:
            content = f.read()
            cubefile = io.StringIO(content.decode('utf-8'))
            
        else:
            raise Exception("File was not extracted correctly")
        cube = read_cube(cubefile) # Only takes atoms and electron density
        
        n = cube['data']
        a = cube['atoms']
        og = cube['origin'][0:3]
        
        # Make consistent input shape and construct ground_truth
        n, flag = self.clean(a, n, og, max_size=6)
        target = self.ground_truth(a, radius=8)
        
        if flag:
            # Returning the volumetric data with single channel
            return n[np.newaxis, :, :, :], target
        else:
            return None
                
    def __len__(self):
        return len(self.names) 

    def max_distance(self, a):
        pos = a.get_positions()
        dist = 0
        for i in range(3):
            d = np.max( pos[:,i] ) - np.min( pos[:, i]) 
            if d > dist:
                dist = d
        return dist

    def clean(self, a, n, og, max_size=6):
        
        # Tells if the entry should be included in training or not
        flag = self._check_entry(a, max_size=max_size)
        
        if flag:
            n = self._pad_density(n) # Maybe implement loss of electrons
            n = self._center(a, n, og)
            
        return n, flag
    
    def _check_entry(self, a, max_size=6):
        if self.max_distance(a) > max_size:
            return False
        else:
            return True
    
    # ONLY SHIFT ATOMS AS OF NOW!
    def _center(self, a, n, og):
        box_center = np.array([0, 0, 0,])
        
        #Finding molecule center (geometric)
        mol_center = np.zeros(3)
        
        pos = a.get_positions()
        for i in range(3):
            d = pos[:,i].max() - pos[:, i].min()
            atoms_index = np.argmin(pos[:, i])
            edge_atom = a[atoms_index]
            
            mol_center[i] = edge_atom.position[i]+d/2
        
        #Translating the positions
        p = box_center - mol_center
        a.positions = a.positions + p    

        #Shifting electron density
        p_n =  np.round(p*20).astype(int)
        p_o = np.round( (a.cell.lengths()+2*og)*20/2 ).astype(int) # Shift due to origin
        return shift(n, p_n+p_o)

            
    def _pad_density(self, n):
        mx = self.input_grid
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
    

    
    def ground_truth(self, a, radius=8):
        X, Y, Z = np.ogrid[:self.input_grid, :self.input_grid, :self.input_grid]
        true = np.zeros((self.input_grid, self.input_grid, self.input_grid))
        
        atoms_pos = np.round(a.get_positions()*20).astype(int) + int(self.input_grid/2)
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
