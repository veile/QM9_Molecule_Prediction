#import torch
import argparse
import sys

from training_procedure import TrainingProcedure
from unet import Net
#from dataset import MolecularDataset
from prepare_set import choose_set

from os import path, mkdir
from torchsummary import summary    

if __name__ == '__main__':
    # Using argparse to control the training directly from command line
    parser = argparse.ArgumentParser(description='Starts training procedure')
    
    parser.add_argument('path', metavar='-p', type=str, nargs='+',
                    help='Specify folder path for saving files')
    
    parser.add_argument('t', metavar='-t', type=float,
                        help='Training time in seconds')
    
    parser.add_argument('v', metavar='-v', type=float,
                        help='Interval time between validations')
    
    parser.add_argument('s', metavar='-s', type=int, default = 0,
                        help='Number of molecules for validation')
    
    parser.add_argument('--d', type=int, default = 0, dest='dataset',
                        help='Dataset - 0:single, 1:small_set, 2:full')
    
    parser.add_argument('--i', type=int, default = 0, dest='index',
                        help='Index to choose single molecule')
    
    args = parser.parse_args()

    dataset, grid = choose_set(args.dataset, single_idx=args.index)
    
    # Inititates the nerual network
    net = Net(8)
    PATH = args.path[0]
    if PATH[-1] != "/":
        PATH = PATH + "/"
        
    # Checks if it should load saved state or continue as new
    if path.exists(PATH):
        with open(PATH+"specs.txt", 'r') as f:
            content = f.read().splitlines()
            test_size = int(content[1])
            dataset_idx = int(content[2])
            
        if dataset_idx != args.dataset:
            raise Exception("Another dataset has been\
                            used to train this network!")
            
        train = TrainingProcedure.load_state(net, dataset,
                                             test_size=test_size,
                                             PATH=PATH)
    else:
        mkdir(PATH)
        
        # Creates file with all specifications
        # Should maybe be implemented in TrainingProcedure
        orig_stdout = sys.stdout
        with open(PATH+"specs.txt", 'w') as f:
            sys.stdout = f
            f.write(PATH)
            f.write("\n")
            f.write(str(args.s))
            f.write("\n")
            f.write(str(args.dataset))
            f.write("\n")
            f.write(str(len(dataset)))
            f.write("\n")
            summary(net.cuda(), (1, grid, grid, grid))
        sys.stdout = orig_stdout
        
        train = TrainingProcedure(net, dataset, test_size=args.s, 
                                  PATH=PATH)
        train.set_optimizer()
        train.set_criterion()
        
    train.run(args.t, args.v)