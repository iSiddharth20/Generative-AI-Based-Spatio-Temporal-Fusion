'''
Module for Loss Functions :
    - Maximum Entropy Principle (MEP)
    - Maximum Likelihood Principle (MLP)
    - Structural Similarity Index Measure (SSIM)
'''

# Import Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

'''
Class for Composite Loss with MaxEnt Regularization Term
    - Maximum Entropy Principle
'''
class LossMEP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMEP, self).__init__()
        self.alpha = alpha  # Weighting factor for the loss

    def forward(self, output, target):
        mse_loss = F.mse_loss(output, target)
        # Assume output to be raw logits: calculate log_probs and use it to compute entropy
        log_probs = F.log_softmax(output, dim=1)  # dim 1 is the channel dimension
        probs = torch.exp(log_probs)
        entropy_loss = -torch.sum(probs * log_probs, dim=1).mean()
        
        # Combine MSE with entropy loss scaled by alpha factor
        composite_loss = (1 - self.alpha) * mse_loss + self.alpha * entropy_loss
        return composite_loss

'''
Class for Mean Squared Error (MSE) Loss
    - Maximum Likelihood Principle
'''
class LossMSE(nn.Module):
    def forward(self, output, target):
        print('Executing forward of LossMSE Class from losses.py')
        likelihood_loss = F.mse_loss(output, target)  # Compute MSE loss using functional API
        return likelihood_loss

'''
Class for Structural Similarity Index Measure (SSIM) Loss
    - Maximum Likelihood Principle
    - In PyTorch, loss is minimized, by doing 1 - SSIM, minimizing the loss function will lead to maximization of SSIM
'''
class SSIMLoss(nn.Module):
    def __init__(self, data_range=1, size_average=True):
        super(SSIMLoss, self).__init__()
        # Initialize SSIM module
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average)

    def forward(self, img1, img2):
        print('Executing forward of SSIMLoss Class from losses.py')
        ssim_value = self.ssim_module(img1, img2)  # Compute SSIM
        return 1 - ssim_value  # Return loss
