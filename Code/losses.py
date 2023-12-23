'''
Module for Loss Functions :
    - Maximum Entropy Principle (MEP)
    - Maximum Likelihood Principle (MLP)
    - Structural Similarity Index Measure (SSIM)
'''

# Import Necessary Libraries
import torch
import torch.nn as nn
from pytorch_msssim import SSIM

'''
Class for Composite Loss with MaxEnt Regularization Term
    - Maximum Entropy Principle
'''
class LossMEP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMEP, self).__init__()
        self.alpha = alpha  # Weighting factor for the loss
        self.mse = nn.MSELoss()  # Mean Squared Error loss

    def forward(self, output, target):
        mse_loss = self.mse(output, target)  # Compute MSE Loss
        entropy = -torch.sum(target * torch.log(output + 1e-8), dim=-1).mean() # Compute Entropy
        composite_loss = self.alpha * mse_loss + (1 - self.alpha) * entropy # Compute Composite Loss
        return composite_loss

'''
Class for Mean Squared Error (MSE) Loss
    - Maximum Likelihood Principle
'''
class LossMSE(nn.Module):
    def __init__(self):
        super(LossMSE, self).__init__()
        self.mse = nn.MSELoss()  # Mean Squared Error loss

    def forward(self, output, target):
        likelihood_loss = self.mse(output, target)  # Compute MSE loss
        return likelihood_loss

'''
Class for Structural Similarity Index Measure (SSIM) Loss
    - Maximum Likelihood Principle
    - In PyTorch, loss is minimized, by doing 1 - SSIM, minimizing the loss function will lead to maximization of SSIM
'''
class SSIMLoss(nn.Module):
    def __init__(self, data_range=1, size_average=True):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range  # The range of the input image (usually 1.0 or 255)
        self.size_average = size_average  # If True, the SSIM of all windows are averaged
        # Initialize SSIM module
        self.ssim_module = SSIM(data_range=self.data_range, size_average=self.size_average)

    def forward(self, img1, img2):
        ssim_value = self.ssim_module(img1, img2)  # Compute SSIM
        return 1 - ssim_value  # Return loss