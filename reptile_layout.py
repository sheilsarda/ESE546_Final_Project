import torch 
import torch.nn as nn
import torch.optim as optim

# Data Wrangling
import numpy as np
import random

class Reptile(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(Reptile, self): 
        params = np.ones(output_dim) # maybe make Gaussian?
        optimizer = optim.SGD(params, lr=self.lr_init)
        self.lr_init = 1e-3
    
    def train(params_init, tasks, model, data, k):

        '''
        k steps of SGD or Adam
        '''

        all_params = params_init

        for _ in iterations:

            task = sample(tasks, 1)

            # Need some stuff from the tasks

            for _ in range(k):

                model.train()

                # Forward Pass
                outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()

                # Training Loss
                loss.backward()
                optimizer.step()

                # Forward Pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()

                # Training Loss
                loss.backward()
                optimizer.step()

            single_task_params = optimizer.params
            
            all_params = all_params + epsilon*(single_task_params - all_params)

            # epsilon = self.lr_init ???????

            '''
            In the last step, instead of simply updating φ in the direction φe − φ, we can treat (φ − φe) as a
            gradient and plug it into an adaptive algorithm such as Adam [10]. (Actually, as we will discuss in
            Section 5.1, it is most natural to define the Reptile gradient as (φ − φe)/α, where α is the stepsize
            '''

    
    def sample(tasks, quantity):

        '''
        Returns a number of sampled tasks
        '''

        task = random.choices(tasks, k=quantity)
        return task

    def compute_loss():
        pass