# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:42:09 2020

@author: Thomas
"""
from mayavi import mlab
import torch
from dataset import MolecularDataset, collate_none

data_dir = "Data/"
dataset =  MolecularDataset(data_dir)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)



n = next(iter(loader))[0].numpy().reshape(200,200,200)
mlab.contour3d(n, contours=5, transparent=True)

#
#def electron3d(dataset, n):
#    values = dataset[n]['data']
#
#
#    print("Looking at " + str(dataset[n].atoms.symbols) )
#    mlab.contour3d(values, contours=50, transparent=True, vmin=-0.1)#, extent=[0, 198, 0, 198, 0, 198])
##    mlab.pipeline.volume(mlab.pipeline.scalar_field(values))
#
#
#
#electron3d(dataset, 5)