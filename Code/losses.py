'''
Module for Loss Functions 
    - Maximum Entropy Principle (MEP)
    - Maximum Likelihood Principle (MLP)
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a class for the Maximum Entropy Principle (MEP) Loss
class LossMEP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMEP, self).__init__()
        # Regularization Parameter Weight 
        self.alpha = alpha  
        # Base Loss Function (MSE)
        self.mse = nn.MSELoss() 

    def forward(self, output, target):
        # Compute the MSE loss
        mse_loss = self.mse(output, target)  
        # Compute Entropy of the Target Distribution
        entropy = -torch.sum(target * torch.log(output + 1e-8), dim=-1).mean()
        # Compute Composite Loss Function with MaxEnt Regularization Term
        regularized_loss = self.alpha * mse_loss + (1 - self.alpha) * entropy
        # Return Composite Loss 
        return regularized_loss  

# Define a class for the Maximum Likelihood Principle (MLP) Loss
class LossMLP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMLP, self).__init__()
        # Regularization Parameter Weight
        self.alpha = alpha
        # Mean Squared Error Loss
        self.mse = nn.MSELoss()  

    def forward(self, output, target):
        # Compute the MSE loss
        likelihood_loss = self.mse(output, target)  
        # Compute Loss Function with Maximum Likelihood Principle
        regularized_loss = self.alpha * likelihood_loss
        # Return Loss
        return regularized_loss