from dataset import MolecularDataset, collate_none

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

class TrainingProcedure():
    def __init__(self, net, dataset, test_size):
        self.best_error = None
        self.steps = 0
        self.validations = 0

        train_size = len(dataset) - test_size
        
        torch.manual_seed(42)
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size]) 
        torch.manual_seed(np.random.randint(100000))

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

        #Using CUDA if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Compute device: ")
        print(self.device)
        
        self.net = net
        self.net.to(self.device)
    
    @classmethod
    def load_state(cls, net, dataset, test_size):
        cls = cls(net, dataset, test_size)
       
        # Loading the rest from saved file
        latest_state = torch.load("./TrainingProcedure.pth")
        cls.best_error = latest_state["BestError"]
        cls.steps = latest_state["Steps"]
        cls.validations = latest_state["Validations"]
        
        cls.net.load_state_dict(latest_state["Model"])
        
        cls.set_optimizer()
        cls.set_criterion()
        cls.optimizer.load_state_dict(latest_state["Optimizer"])
        
        return cls
     
    def set_optimizer(self, algorithm='Adam', lr=1e-4, momentum=0):
        if algorithm == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        elif algorithm == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        else:
            raise Exception("No optimizer algorithm with name %s" %algorithm)
    
    def set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        
    def run(self, max_runtime, validation_interval=3600):
        runtime = time.time()
        it = iter(self.train_loader)
        
        train_time = time.time()
        while (time.time() - runtime) < max_runtime:
            # Loading the next entries in the iterator
            try:
                inputs, targets = next(it)
            except StopIteration:
                print("Got through the whole dataset - Reloading the iterator")
                it = iter(self.train_loader)
                pass
            except:
                print("Something went wrong with loading the next entry")
             
            self.step(inputs, targets)
            
            
            if (time.time()-train_time) > validation_interval:
                self.validate()
                train_time = time.time()
        self.validate()
        print("Finished training with %i molecules" %self.steps) 
            
            
    def step(self, inputs, targets):
        # Moving the tensors to device
        inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
        
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
       
        self.steps += 1
        print('[%d]      %.5f' %(self.steps, loss.item()))
    
    def validate(self):
        start = time.time()
        
        validation_error = []
        #Testing the model
        for inputs, targets in self.test_loader:
            # Moving the tensors to device
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
            
            outputs = self.net(inputs)
            validation_error = np.append(validation_error, self.criterion(outputs, targets).cpu().detach().numpy() )
        validation_error = validation_error.mean()
        print("Validation took:")
        print(time.time()-start)
        print("Validation error after %i molecules is: %.5f" %(self.steps, validation_error ))
        
        self.validations += 1
        self.save_state(validation_error)
        
    def save_state(self, validation_error):
        with open("validation_errors.txt", 'a') as f:
            f.write( str(self.steps)+" "+str(validation_error)+"\n")
            
        if self.best_error is None:
            self.best_error = validation_error
            
        elif self.best_error >  validation_error:
            self.best_error = validation_error
            torch.save({'Model': self.net.state_dict()}, "./best_model.pth")
            
        torch.save({'Model': self.net.state_dict(),
                    'Optimizer': self.optimizer.state_dict(),
                    'Steps': self.steps,
                    'BestError': self.best_error,
                    'Validations': self.validations}, "./TrainingProcedure.pth")