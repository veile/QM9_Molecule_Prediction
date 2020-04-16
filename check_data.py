from dataset import MolecularDataset
from vol_render import volume
import numpy as np

import tarfile
import io
from ase.io.cube import read_cube

tar_filename = "qm9_000xxx_29.cube.tar.gz"


## =============================================================================
## Before clean
## =============================================================================
#tar = tarfile.open(tar_filename, "r:gz")
#
## Number of files
#names = tar.getnames()
#names.sort()
#names = np.array(names)   
#
#N_sum = 0
#for name in names:
#    entry = tar.getmember( name )
#    f = tar.extractfile(entry)
#    
#    content = f.read()
#    cubefile = io.StringIO(content.decode('utf-8'))
#    cube = read_cube(cubefile) # Only takes atoms and electron density
#
#    n = cube['data']
#    print(cube['atoms'])
#    print(cube['origin'])
#    N_sum += n.sum()*0.05**3
#print(N_sum)



# =============================================================================
# After clean
# =============================================================================
#dataset =  MolecularDataset(tar_filename, input_grid=160)
#N_sum = 0
#for i in range(len(dataset)):
#    inputs, targets = dataset[i]
#    N = inputs.sum()*0.05**3
#    print(N)
#    N_sum += N
#print(N_sum)