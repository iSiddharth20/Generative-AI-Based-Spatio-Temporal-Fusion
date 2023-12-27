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
        mse_loss = F.mse_loss(output, target)  # Compute MSE Loss using functional API
        # Normalize the output tensor along the last dimension to represent probabilities
        output_normalized = torch.softmax(output, dim=-1)
        # Compute Entropy
        entropy = -torch.sum(target * torch.log(output_normalized + 1e-8), dim=-1).mean()
        # Compute Composite Loss
        composite_loss = self.alpha * mse_loss + (1 - self.alpha) * entropy
        return composite_loss

'''
Class for Mean Squared Error (MSE) Loss
    - Maximum Likelihood Principle
'''
class LossMSE(nn.Module):
    def forward(self, output, target):
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
        ssim_value = self.ssim_module(img1, img2)  # Compute SSIM
        return 1 - ssim_value  # Return loss
