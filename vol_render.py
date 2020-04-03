# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:38:57 2020

@author: s153012
"""

import numpy as np
import math
from dataset import MolecularDataset
import torch
import pyvista as pv

def pad_density(n, mx=200):
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
    
def volume(n, name):
#    N = np.shape(n)[0]
#    lim = N*0.05/2
#    X, Y, Z = np.mgrid[-lim:lim:N*1j, -lim:lim:N*1j, -lim:lim:N*1j] 
#    
#    # Data
#    grid = pv.StructuredGrid(X, Y, Z)
#    grid["vol"] = n.flatten()
    # contours = grid.contour([n.mean()])

    # Visualization
    pv.set_plot_theme('document')
    p = pv.Plotter()
    p.add_volume(n)
    p.show_grid()
    # p.add_mesh(contours, scalars=contours.points[:, 2], show_scalar_bar=False)
    p.show(screenshot='Volumes/%s.pdf' %name)
    # p.show()
    
def volume2(n1, n2):
#    N = np.shape(n)[0]
#    lim = N*0.05/2
#    X, Y, Z = np.mgrid[-lim:lim:N*1j, -lim:lim:N*1j, -lim:lim:N*1j] 
#    
#    # Data
#    grid = pv.StructuredGrid(X, Y, Z)
#    grid["vol"] = n.flatten()
    # contours = grid.contour([n.mean()])

    # Visualization
    pv.set_plot_theme('document')
    p = pv.Plotter()
    p.add_volume(n1)
    p.add_volume(n2)
    p.show_grid()
    # p.add_mesh(contours, scalars=contours.points[:, 2], show_scalar_bar=False)
    # p.show(screenshot='Volumes/%s.pdf' %name)
    p.show()



def test_volume():
    X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
    values = np.sin(X*Y*Z) / (X*Y*Z)
    volume(values)

def load(index):
    tarfile = "./qm9_000xxx_29.cube.tar.gz"
    dataset =  MolecularDataset(tarfile)
    return dataset[index]

if __name__ == "__main__":
    # n, gt = load(0)
    # n = np.squeeze(n)
   
    # volume(n, "density")
    # volume(gt, "ground_truth")


    
    filename = "Results/160input_1000.npz"
    data = np.load(filename)
    gt = data['gt']
    out = np.squeeze( data['out'] )
    
    reconstruct = np.dot(out.T, np.arange(6) )
    volume2(gt, reconstruct)
    # volume(gt, "ground_truth")
    # volume(reconstruct, "reconstruction")
