import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data.dataloader import default_collate

def collate_none(batch):
    """
    Collate function for the PyTorch DataLoader that filters out None output,
    which occurs when the molecule is too big. If no molecule in the batch is
    within the size criteria, the function return None, which has to be handled
    client side.
    """
    # Filters out molecules that does not fit the size criteria
    batch = list( filter (lambda x:x is not None, batch) )
    
    # Makes it possible to return both n, target and atoms
    N = len(batch)
    n = [item[0] for item in batch]
    target = [item[1] for item in batch]
    atoms = [item[2] for item in batch]
    batch = [ (n[i], target[i]) for i in range(N)]
    
    if np.size(batch) > 0:
        return default_collate(batch), atoms
    else:
        return None

class TrainingProcedure():
    """
    Object that trains a neural network from a given dataset
    
    Parameters
    ----------
    net : torch.nn.Module
        The neural network that should be optimized - should inherit from 
        torch.nn.Module.
    dataset : torch.utils.data.Dataset
        Instance of a pytorch dataset
    test_size : int
        Number of entries in the dataset that should be used for validation
    PATH : String
        String containin the path to where the neural network should be saved
    """
    def __init__(self, net, dataset, test_size, PATH):
        # Defaults the PATH structure
        if PATH[-1] == '/':
            self.PATH = PATH
        else:
            self.PATH = PATH + '/'
        
        # Important values to follow during training
        self.best_error = None
        self.steps = 0
        self.validations = 0


        # Splits the dataset
        # A seed is used such that it is possible to access the test set
        # after the training is done.
        train_size = len(dataset) - test_size
        
        torch.manual_seed(42)
        train_set, test_set = torch.utils.data.random_split(dataset,
                                            [train_size, test_size]) 
        torch.manual_seed(np.random.randint(100000))

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=1,
                                                    num_workers=0,
                                                    shuffle=True,
                                                    collate_fn=collate_none)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=1,
                                                    num_workers=0,
                                                    shuffle=False, 
                                                    collate_fn=collate_none)

        # Checks if the dataset is a single molecule
        if len(dataset) == 1:
            self.single = True
        else:
            self.single = False        
        
        #Using CUDA if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available()\
                                            else "cpu")
        print("Compute device: ")
        print(self.device)
        
        self.net = net
        self.net.to(self.device)
    
    @classmethod
    def load_state(cls, net, dataset, test_size, PATH):
        """
        Loads a previously saved training procedure. Takes the same input
        as __init__.
        """
        print("Loading previous model from %s" %PATH)
        cls = cls(net, dataset, test_size, PATH)
       
        # Loading the rest from saved file
        latest_state = torch.load(cls.PATH+"TrainingProcedure.pth")
        cls.best_error = latest_state["BestError"]
        cls.steps = latest_state["Steps"]
        cls.validations = latest_state["Validations"]
        
        cls.net.load_state_dict(latest_state["Model"])
        
        cls.set_optimizer()
        cls.set_criterion()
        cls.optimizer.load_state_dict(latest_state["Optimizer"])
        
        return cls
     
    def set_optimizer(self, algorithm='Adam', lr=1e-4, momentum=0):
        """
        Choose optimizer between Adaptive momentum and Stochastic gradient 
        descent. Defaults to the ADAM optimizer with a learning rate of 1e-4
        """
        if algorithm == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        elif algorithm == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
                                       momentum=momentum)
        else:
            raise Exception("No optimizer algorithm with name %s" %algorithm)
    
    def set_criterion(self, criterion=None):
        """
        Choose loss function used in the optimizer. Defaults to
        CrossEntropyLoss.
        """
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
    
    def next_entry(self):
        """
        Loads the next entry in the train_set. Handles if None is returned
        by the iterator. If 20 molecules are too big in a row, the training
        procedure will terminate.
        """
        i = 0    # i is the counter
        while i < 20:
            try:
                tensors, _ = next(self.it)
                inputs, targets = tensors
                return inputs, targets
            
            except StopIteration:
                print("Got through the whole dataset - Reloading the iterator")
                self.it = iter(self.train_loader)
                pass
            
            except TypeError:
                print("Molecule is too big")
                pass
            i+=1
                
        raise Exception("Did not succeed to load next entry after 10 tries")
    
    def run(self, max_runtime, validation_interval=3600):
        """
        Method that starts the training on the neural network. It will save 
        its last statte before terminating.
        
        Parameters
        ----------
        max_runtime : float
            Time in seconds the neural network should be trained
        validation_interval : float
            Time in seconds between calculating the validation errors
        """
        self.it = iter(self.train_loader)
        
        if self.single:
            self.single_run(epochs=1000)
            
        else:
            runtime = time.time()
            running_loss=0
            
            train_time = time.time()
            while (time.time() - runtime) < max_runtime:
                inputs, targets = self.next_entry()
                running_loss += self.step(inputs, targets)
                
                # Prints training error after M steps
                M = 50
                if self.steps % M == (M-1):
                    print('[%d]      %.5f' %(self.steps, running_loss/M))
                    running_loss = 0
                
                if (time.time()-train_time) > validation_interval:
                    self.validate()
                    train_time = time.time()
                    
            self.validate()
            print("Finished training with %i molecules" %self.steps) 
            
    def single_run(self, epochs):
        """
        Distinct method for training on single molecules to speed up training        
        """        
        inputs, targets = self.next_entry()
          
        for i in range(epochs):
            loss = self.step(inputs, targets)
            print("[%i]     %.5f" %(self.steps, loss))
        self.save_state(loss)
              
          
    def step(self, inputs, targets):
        """
        Takes one step in the optimizer and returns the calculated loss
        """
        # Moving the tensors to device
        inputs, targets = inputs.to(self.device).float(),\
                          targets.to(self.device).long()
        
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
       
        self.steps += 1
        
        return loss.item()
    
    def validate(self):
        """
        Calculates the validation error for the molecules in the test set
        """
        start = time.time()
        val_filtered = 0
        
        validation_error = []
        #Testing the model
        for items in self.test_loader:
            try:
                tensors, _ = items
                inputs, targets = tensors
            except TypeError:
                val_filtered += 1
                continue
            # Moving the tensors to device
            inputs, targets = inputs.to(self.device).float(),\
                              targets.to(self.device).long()
            
            outputs = self.net(inputs)
            validation_error = np.append(validation_error,
                                         self.criterion(outputs, targets)\
                                         .cpu().detach().numpy() )
        validation_error = np.mean(validation_error)
        print("Validation took:")
        print(time.time()-start)
        print("Validation set contained %i molecules that were too big"\
              %val_filtered)
        print("Validation error after %i molecules is: %.5f"\
              %(self.steps, validation_error ))
        
        self.validations += 1
        self.save_state(validation_error)
        
    def save_state(self, validation_error):
        """
        Saves the current state of the training procedure and if lowest
        validation error is achieved, the model is saved.
        """
        with open(self.PATH+"validation_errors.txt", 'a') as f:
            f.write( str(self.steps)+" "+str(validation_error)+"\n")
            
        if self.best_error is None:
            self.best_error = validation_error
            
        elif self.best_error >  validation_error:
            self.best_error = validation_error
            torch.save({'Model': self.net.state_dict()},
                        self.PATH+"best_model.pth")
            
        torch.save({'Model': self.net.state_dict(),
                    'Optimizer': self.optimizer.state_dict(),
                    'Steps': self.steps,
                    'BestError': self.best_error,
                    'Validations': self.validations},
                    self.PATH+"TrainingProcedure.pth")