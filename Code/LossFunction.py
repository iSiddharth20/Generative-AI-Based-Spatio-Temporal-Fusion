'''
Module that specifies Composite Loss Function and Maximum Entropy Calculation (MaxEnt)
--------------------------------------------------------------------------------
This is a Template Code and Needs to be Modified based on the Problem Statement 
'''


# Importing Necessary Libraries
import torch
import torch.nn as nn
import numpy as np

# Defining the MaxEnt Class
class MaxEnt(): 
    def __init__(self):
        super(MaxEnt, self).__init__()

    def forward(self, pred):
        # pred is the mean of the Gaussian distribution
        batch_size = pred.size(0)
        # Calculate variance along batch dimension
        variance = pred.var(dim=0) + 1e-12  # Avoid division by zero
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * variance).sum()
        average_entropy = entropy / batch_size
        # Multiply by -1 so that higher entropy gives a smaller loss
        return -average_entropy
    
# Defining the Composite Loss Function (MSE + MaxEnt)
class CompositeLossFunction(nn.Module):
    def __init__(self, alpha=0.5):
        super(CompositeLossFunction, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.maxent_loss = MaxEnt()
        self.alpha = alpha

    def forward(self, pred, target):
        # Mean Squared Error component
        mse = self.mse_loss(pred, target)
        # MaxEnt regularization component using the MaxEnt Class
        maxent = self.maxent_loss(pred)
        # Combine the loss components
        composite_loss = (1 - self.alpha) * mse + self.alpha * maxent
        return composite_loss
    
# Defining the Regular Loss Function (MSE only)
class RegularLossFunction(nn.Module):
    def __init__(self, alpha=0.5):
        super(CompositeLossFunction, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.alpha = alpha

    def forward(self, pred, target):
        # Mean Squared Error component
        mse = self.mse_loss(pred, target)
        # Combine the loss components
        composite_loss = (1 - self.alpha) * mse + self.alpha
        return composite_loss
    
