from training_procedure import TrainingProcedure
from unet import Net
from dataset import MolecularDataset

import torch
from os import path





tarfile = "qm9_000xxx_29.cube.tar.gz"
dataset =  MolecularDataset(tarfile)
net = Net(8)

if path.exists("./TrainingProcedure.pth"):
    train = TrainingProcedure.load_state(net, dataset, test_size=2)
else:
    train = TrainingProcedure(net, dataset, test_size=2)
    train.set_optimizer()
    train.set_criterion()

"""
if path.exists("./TrainingProcedure.pth"):
    train = TrainingProcedure.load_state(tarfile, test_size=2)
else:
    train = TrainingProcedure(tarfile, test_size=2, neurons=8)
    train.set_optimizer()
    train.set_criterion()
"""
train.run(60, 20)