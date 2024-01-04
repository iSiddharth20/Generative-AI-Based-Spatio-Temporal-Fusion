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
Class for Composite Loss with Maximum Entropy Principle Regularization Term
'''
class LossMEP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMEP, self).__init__()
        self.alpha = alpha  # Weighting factor for total variation loss
        
    def forward(self, output, target):
        mse_loss = F.mse_loss(output, target)  # MSE loss between output and target
        # Calculate dimensions of the output
        batch_size, _, height, width = output.size()
        # Total variation loss for output, penalizes large differences between neighboring pixel-values
        tv_loss = torch.sum(torch.abs(output[:, :, :-1] - output[:, :, 1:])) + \
                  torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:]))
        tv_loss /= batch_size * height * width  # Normalize by total size
        # Composite loss
        loss = mse_loss + self.alpha * tv_loss
        # Return the composite loss
        return loss  

'''
Class for Mean Squared Error (MSE) Loss
'''
class LossMSE(nn.Module):
    def forward(self, output, target):
        likelihood_loss = F.mse_loss(output, target)  # Compute MSE loss using functional API
        return likelihood_loss

'''
Class for Composite Loss with Structural Similarity Index Measure (SSIM) Loss
    - In PyTorch, loss is minimized, by doing 1 - SSIM, minimizing the loss function will lead to maximization of SSIM
'''
class SSIMLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SSIMLoss, self).__init__()
        self.alpha = alpha
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=1)

    def forward(self, seq1, seq2):
        N, T = seq1.shape[:2]
        ssim_values = []
        mse_values = []
        for i in range(N):
           for t in range(T):
            seq1_slice = seq1[i, t:t+1, ...] 
            seq2_slice = seq2[i, t:t+1, ...]
            ssim_val = self.ssim_module(seq1_slice, seq2_slice)
            mse_val = F.mse_loss(seq1_slice, seq2_slice)
            ssim_values.append(ssim_val) # Compute SSIM for each frame in the sequence
            mse_values.append(mse_val) # Compute MSE for each frame in the sequence
        avg_ssim = torch.stack(ssim_values).mean() # Average SSIM across all frames
        avg_mse = torch.stack(mse_values).mean() # Average MSE across all frames
        combined_loss = (1 - self.alpha) * avg_mse + self.alpha * (1 - avg_ssim)  # SSIM is maximized, so we subtract from 1
        return combined_loss
