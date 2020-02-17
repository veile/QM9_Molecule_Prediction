from dataset import MolecularDataset
import numpy as np
from mayavi import mlab

data_dir = "Data/"
dataset = MolecularDataset(data_dir)

N_before = np.sum( dataset.no_electrons() )

dataset.clean()

N_after = np.sum( dataset.no_electrons() )

print("Number of electrons before cut:_ %.5f" %N_before)
print("Number of electrons after cut: %.5f" %N_after)
print("Total removed electrons: %.5f" %(N_before - N_after) )

def electron3d(dataset, n):
    values = dataset[n]['data']


    print("Looking at " + str(dataset[n].atoms.symbols) )
    mlab.contour3d(values, contours=50, transparent=True, vmin=-0.1)#, extent=[0, 198, 0, 198, 0, 198])
#    mlab.pipeline.volume(mlab.pipeline.scalar_field(values))
   
electron3d(dataset, 5)
