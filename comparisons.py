#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:14:34 2020

@author: s153012
"""
import numpy as np

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})
from matplotlib.colors import LinearSegmentedColormap
colors = [ (0, 0, 0, 0.2), (1, 1, 1), (0, 0, 0), (1, 0, 0), (0, 0, 1)]
atoms_cmap = LinearSegmentedColormap.from_list('atoms', colors, N=5)

# =============================================================================
# Kernel size and optimizer
# =============================================================================
paths= ["mol17_kernel"+str(i) for i in range(3,7)]
paths.append("mol17_kernel6_SGD")

fig, axes = plt.subplots(2, 3, figsize=(12, 10))

axes = axes.flatten()

titles = ['Kernel size 3\nAdam Optimizer', 'Kernel size 4\nAdam Optimizer',
          'Kernel size 5\nAdam Optimizer', 'Kernel size 6\nAdam Optimizer',
          'Kernel size 6\nSGD Optimizer']

for i, f in enumerate(paths):
    data = np.load(f+"/results.npz")
    out = data['out']
    gt = data['gt']
    
    M = gt.shape[0]
    scaled = int( M/200*88 )
    
#    plc +=1
#    plt.subplot(plc)
    axes[i+1].set_title(titles[i])
    axes[i+1].imshow(np.argmax(out, axis=0)[:, :, scaled], cmap=atoms_cmap,
                     vmin=0, vmax=4)
    axes[i+1].set_xticks([])
    axes[i+1].set_yticks([])
    
    
axes[0].set_title("Ground Truth")
im = axes[0].imshow(gt[:, :, scaled], cmap=atoms_cmap, vmin=0, vmax=4)
axes[0].set_xticks([])
axes[0].set_yticks([])


#ticks = [0.75/2+0.75*i for i in range(5)]
ticks, dh = np.linspace(0,4, 6, retstep=True)
ticks = ticks[1:]-dh/2
cbar = fig.colorbar(im, ax=axes[:], orientation='horizontal',
                    shrink=0.9, pad=0.1, ticks=ticks)
cbar.ax.set_xticklabels(['BG', 'H', 'C', 'O', 'N'])
#fig.tight_layout()
plt.show()
    
# =============================================================================
# Number of levels
# =============================================================================
paths = ["mol17_kernel5", "mol17_2levels"]
titles = ["3 Levels", "2 Levels"]
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes = axes.flatten()

for i, f in enumerate(paths):
    data = np.load(f+"/results.npz")
    out = data['out']
    gt = data['gt']
    
    M = gt.shape[0]
    scaled = int( M/200*88 )
    
#    plc +=1
#    plt.subplot(plc)
    axes[i+1].set_title(titles[i])
    axes[i+1].imshow(np.argmax(out, axis=0)[:, :, scaled], cmap=atoms_cmap,
                     vmin=0, vmax=4)
    axes[i+1].set_xticks([])
    axes[i+1].set_yticks([])
    
    
axes[0].set_title("Ground Truth")
im = axes[0].imshow(gt[:, :, scaled], cmap=atoms_cmap, vmin=0, vmax=4)
axes[0].set_xticks([])
axes[0].set_yticks([])


#ticks = [0.75/2+0.75*i for i in range(5)]
ticks, dh = np.linspace(0,4, 6, retstep=True)
ticks = ticks[1:]-dh/2
cbar = fig.colorbar(im, ax=axes[:], orientation='vertical',
                    shrink=0.5, ticks=ticks)
cbar.ax.set_yticklabels(['BG', 'H', 'C', 'O', 'N'])
#fig.tight_layout()
plt.show()
