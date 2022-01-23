import numpy as np
import math
import tarfile
import io
import zlib
import ase

from ase.io.cube import read_cube
from ase.data import covalent_radii

from torch.utils.data.dataset import Dataset
from scipy.ndimage.interpolation import shift

def extract_zz(tar, entry):
    """
    Extracts cube-files from files with .zz compression
    Made by Peter Bjørn Jørgensen at DTU Energy
    
    Parameters
    ----------
    tar : Instance of tarfile
        Instance should be created as: tarfile.open(tar_filename, 'r')
    entry : tar member
        Which element to unpack in tarfile. Should be created as:
            tarfile.getmember(filename)
            
    Returns
    -------
    n : ndarray pf shape (X, Y, Z)
        Electron density extracted from the cube-file
    atom : ase.Atoms object
        Contains type of atoms, position, cell etc.
    origin : ndarray of shape (3,)
        Shifted origin between electron density and atomsobject
    """
    buf = tar.extractfile(entry)
    cube_file = io.StringIO(zlib.decompress(buf.read()).decode())
    cube = read_cube(cube_file)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    
    # By convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3,
    # so we do the conversion here
    cube["data"] *= (1./ase.units.Bohr**3)
    return cube["data"], cube["atoms"], origin
    
def extract_gz(tar, entry):
    """
    Extracts cube-files from files with .tar.gz compression
    
    Parameters
    ----------
    tar : Instance of tarfile
        Instance should be created as: tarfile.open(tar_filename, 'r:gz')
    tarinfo : tar member
        Which element to unpack in tarfile. Should be created as:
            tarfile.getmember(filename)
            
    Returns
    -------
    n : ndarray pf shape (X, Y, Z)
        Electron density extracted from the cube-file
    atom : ase.Atoms object
        Contains type of atoms, position, cell etc.
    origin : ndarray of shape (3,)
        Shifted origin between electron density and atomsobject
    """
    f = tar.extractfile(entry)
    if f is not None:
        content = f.read()
        cubefile = io.StringIO(content.decode('utf-8'))
    else:
        raise Exception("File was not extracted correctly")
    cube = read_cube(cubefile) # Only takes atoms and electron density
    
    n = cube['data']*(1./ase.units.Bohr**3)
    a = cube['atoms']
    og = cube['origin'][0:3]
    return n, a, og
    
class MolecularDataset(Dataset):
    """
    Molecular dataset that inherits from torch.utils.data.dataset.Dataset
    Can be used with torch.utils.data.DataLoader to read data in batches
    
    Parameters
    ----------
    tar_filename : String
        String containing the path to the tar-file
    input_grid : ínt
        Size that the electron density shoudl be padded/cropped to fit
    output_grid : int
        Size that the ground truth should be cropped to i.e. size
        of the neural network output
    
    """
    def __init__(self, tar_filename, input_grid=170, output_grid=160):    
        
        # Chooses the extracts function based on filename - can easily be
        # extended to work for several compression types
        if tar_filename[-3:] == ".gz":
            self.tar = tarfile.open(tar_filename, "r:gz")
            self.extract = extract_gz
        else:
            self.tar = tarfile.open(tar_filename, "r")    
            self.extract = extract_zz
        
        # Read all filenames in a tar-file
        self.names = self.tar.getnames()
        self.names.sort()
        self.names = np.array(self.names)   
        
        # Setting the grid parameters
        self.input_grid = input_grid
        self.output_grid = output_grid
        print("Dataset initiated: %s" %tar_filename)
                
    def __getitem__(self, index):
        entry = self.tar.getmember( self.names[index] )
        
        n, a, og = self.extract(self.tar, entry)
        
        # Make consistent input shape and construct ground_truth
        n, flag = self.clean(a, n, og, max_size=6)
        
        
        
        # If the molecule is bigger than max_size the function return None
        # This has to be accounted for in the data loading
        if flag:
#            self.latest_atom = a
            # Conly calculates target if atom is within the size criteria
            target = self.ground_truth(a)
            
            # Returning the volumetric data with single channel and the atoms
            # object. This also has to be accounted for in the loading
            return n[np.newaxis, :, :, :], target, a
        else:
            return None
                
    def __len__(self):
        return len(self.names)
    
    def max_distance(self, a):
        """
        Return max distance between two atoms in an ase.Atoms object
        Distance is only x- y- and z-distance
        
        Parameters
        ----------
        a : ase.Atoms
            Atoms in which max distance should be found
            
        Returns
        -------
        dist : float
            Max distance between atoms in the x- y- and z-axes
        """
        pos = a.get_positions()
        dist = 0
        for i in range(3):
            d = np.max( pos[:,i] ) - np.min( pos[:, i]) 
            if d > dist:
                dist = d
        return dist

    def clean(self, a, n, og, max_size):
        
        # Tells if the entry should be included in training or not
        flag = self._check_entry(a, max_size=max_size)
        
        # Only operates on the electron density if the molecule is proper size
        if flag:
            n = self._pad_density(n) # Maybe implement loss of electrons
            n = self._center(a, n, og)
            
        return n, flag
    
    def _check_entry(self, a, max_size):
        if self.max_distance(a) > max_size:
            return False
        else:
            return True
    
    def _center(self, a, n, og):
        """
        Centers the molecule and shift the electron density to fit
        """
        box_center = np.array([0, 0, 0])
        
        #F inding molecule center (geometric in x- y- and z-box)
        mol_center = np.zeros(3)
        
        pos = a.get_positions()
        for i in range(3):
            d = pos[:,i].max() - pos[:, i].min()
            atoms_index = np.argmin(pos[:, i])
            edge_atom = a[atoms_index]
            
            mol_center[i] = edge_atom.position[i]+d/2
        
        # Translating the atom positions
        p = box_center - mol_center
        a.positions = a.positions + p    

        # Shifting electron density same amount
        p_n =  np.round(p*20).astype(int)
        
        # Shift due to origin
        p_o = np.round( (a.cell.lengths()+2*og)*20/2 ).astype(int) 
        return shift(n, p_n+p_o)

            
    def _pad_density(self, n):
        
        # Max size of electron density
        mx = self.input_grid
        x, y, z = np.shape(n)
        
        # Crops all axes bigger than mx
        if x > mx:
            l = (x - mx)/2
            n = n[math.floor(l):(x-math.ceil(l)), :, :]
        if y > mx:
            l = (y - mx)/2
            n = n[:, math.floor(l):(y-math.ceil(l)), :]
        if z > mx:
            l = (z - mx)/2
            n = n[:, :, math.floor(l):(z-math.ceil(l))]

        # Update shape (x, y, z) after cropping
        x, y, z = np.shape(n)
        # Returns the the electron density padded with 0 to have (mx, mx, mx)
        return np.pad(n, ( (math.floor((mx-x)/2), math.ceil((mx-x)/2)),
                           (math.floor((mx-y)/2), math.ceil((mx-y)/2)),
                           (math.floor((mx-z)/2), math.ceil((mx-z)/2)) ),
                            'constant', constant_values=0)
    
    def ground_truth(self, a):
        # Setting dictionary for converting atomic number to channel number
        # HCONF - 1, 2, 3, 4, 5
        atomic_dict = {
                1 : 1,
                6 : 2,
                8 : 3,
                7 : 4,
                9 : 5}
        
        # Creates open multi-dimensional meshgrid
        X, Y, Z = np.ogrid[:self.input_grid, 
                           :self.input_grid,
                           :self.input_grid]
        
        # Initiates the ground truth array
        true = np.zeros((self.input_grid, self.input_grid, self.input_grid))
        
        # Converts atoms positions to index
        atoms_pos = np.round(a.get_positions()*20).astype(int)\
                    + int(self.input_grid/2)
        atomic_numbers = a.get_atomic_numbers()
        
        # Insert spheres of values with radius corresponding to the covalent
        # radius and values corresponding to the atomic dictionary defined
        for i in range(len(a)):
            dist_from_center = np.sqrt((X - atoms_pos[i,0])**2 + \
                                       (Y - atoms_pos[i,1])**2 + \
                                       (Z - atoms_pos[i,2])**2)
            
            # Covalent radius is converted from bohr radius to angstrom
            mask = dist_from_center <= int( covalent_radii[ atomic_numbers[i]]\
                                           *0.529177*20 )
            
            true = true + mask*atomic_dict[atomic_numbers[i] ]
       
        # Crops the ground truth such that it fits the wanted output size
        mid = int(self.input_grid/2)
        ds = self.output_grid/2

        true = true[(mid-math.floor(ds)):(mid+math.ceil(ds))
                   ,(mid-math.floor(ds)):(mid+math.ceil(ds))
                   ,(mid-math.floor(ds)):(mid+math.ceil(ds)) ]                    
        return true