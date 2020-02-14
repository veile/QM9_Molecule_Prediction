from dataset import MolecularDataset
import numpy as np

data_dir = "Data/"
dataset = MolecularDataset(data_dir)

N_before = np.sum( dataset.no_electrons() )

dataset.clean()

N_after = np.sum( dataset.no_electrons() )

print("Number of electrons before cut:_ %.5f" %N_before)
print("Number of electrons after cut: %.5f" %N_after)
print("Total removed electrons: %.5f" %(N_before - N_after) )
