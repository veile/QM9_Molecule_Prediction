#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:47:02 2020

@author: s153012
"""
import torch
from dataset import MolecularDataset

def single(single_idx):
    tarfile = "qm9_000xxx_29.cube.tar.gz"
    dataset =  MolecularDataset(tarfile)
    return torch.utils.data.Subset(dataset, [single_idx]), dataset.input_grid

def small(single_idx):
    tarfile = "qm9_000xxx_29.cube.tar.gz"
    dataset =  MolecularDataset(tarfile)
    return dataset, dataset.input_grid

def full(single_idx):
    # Adds all tar-files in the work-dir to a list
    work_dir = "/home/scratch3/pbjo/cube/"
    with open(work_dir+"qm9_all.txt", "r") as f:
        content = f.read()
        tarfiles = content.splitlines()
    
    # Create list of datasets
    datasets = [MolecularDataset(work_dir+x) for x in tarfiles]
    
    # Using PyTorch's implemented function to concatenate all the datasets
    return torch.utils.data.ConcatDataset(datasets), datasets[0].input_grid

def choose_set(index, single_idx = 0):
    """
    Choose what dataset to prepare
    
    Parameters
    ----------
    index : int
        Which dataset to load
        0 - Single molecule in 28 set. Possible to further choose which by
            specifying single_idx
        1 - 28 molecule set
        2 - full dataset
    single_idx : int
        Index for when choosing a single atom
        
    Returns
    ----------
    dataset : torch.utils.data.Dataset
        Returns a molecular dataset
    input_grid : int
        Returns input grid size
    """
    switcher =  {0: single,
                 1: small,
                 2: full}
    func = switcher.get(index)
    
    try:
        return func(single_idx)
    except TypeError:
        print("Dataset is not implemented - please choose from the following")
        print(switcher)
        raise Exception("Not implemented")

if __name__ == '__main__':
    sets, grid = choose_set(1)
    print(len(sets))
    print(grid)
