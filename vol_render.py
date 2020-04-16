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
from matplotlib.colors import ListedColormap

# N = 6
# vals = np.ones((N, 4))
# vals[0] = np.array([255, 255, 255, 0]) # White
# vals[1] = np.array([255, 255, 204, 256]) # Beige (H)
# vals[2] = np.array([0, 0, 0, 256]) # Black (C)
# vals[3] = np.array([255, 0, 0, 256]) # Red (O)
# vals[5] = np.array([0, 0, 255, 256]) # Blue (N)
# vals[4] = np.array([255, 128, 0, 256]) # Orange (F)

N = 256
vals = np.ones((N, 4))
vals[0:43] = np.array([255, 255, 255, 0]) # White
vals[43:86] = np.array([255, 255, 204, 256]) # Beige
vals[86:129] = np.array([0, 0, 0, 256]) # Black
vals[129:172] = np.array([255, 0, 0, 256]) # Red
vals[172:215] = np.array([0, 0, 255, 256]) # Blue 
vals[215:] = np.array([255, 128, 0, 256]) # Orange 

atoms_cmap = ListedColormap(vals/256)

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
    # Camera position
    cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
    
    pv.set_plot_theme('document')
    p = pv.Plotter()
    annotations  = {0: 'Back',
                    1: 'H',
                    2: 'C',
                    3: 'O',
                    4: 'N',
                    5: 'F'
                    }
    sargs = dict(n_labels=6, label_font_size=None, color=None, height=None, position_x=None, position_y=None, vertical=True, interactive=True, fmt=None, use_opacity=False, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False)
    # p.add_volume(np.round(n), clim=[-0.5, np.round(n).max()+0.5], cmap=atoms_cmap, categories=True, multi_colors=True, scalar_bar_args=sargs, annotations=annotations)
    p.add_volume(n)#, cmap=atoms_cmap)
    p.show_grid()
    # p.save_graphic('Volumes/%s.pdf' %name, title='PyVista Export', raster=False, painter=False)
    # p.show()
    p.view_zy(True)
    p.show(screenshot='Volumes/%s' %name)#, cpos=cpos)

    
def volume2(n1, n2):
    # Visualization
    # cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
    pv.set_plot_theme('document')
    p = pv.Plotter(shape=(1,2))
    
    p.add_volume(n1, show_scalar_bar=False)#, cmap=atoms_cmap, categories=True)
    p.show_grid()
    p.view_xy(True)
    
    p.subplot(0,1)
    sargs = dict(n_labels=6, label_font_size=None, color=None, height=None, position_x=None, position_y=None, vertical=True, interactive=False, fmt=None, use_opacity=False, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False)
    p.add_volume(n2, scalar_bar_args=sargs)#, cmap=atoms_cmap)
    p.show_grid()
    p.view_xy(True)
    # p.link_views()
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


    name = "kernel4_1000"
    filename = "Results/%s.npz" %name
    data = np.load(filename)
    gt = data['gt']
    out = np.squeeze( data['out'] )
    
    reconstruct = np.dot(out.T, np.arange(6) ).T
    
    volume2(gt, reconstruct)
    # volume(gt, "%s_gt" %name)
    # volume(reconstruct, "%s_re" %name)
