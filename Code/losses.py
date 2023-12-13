'''
Module that specifies Loss Functions
--------------------------------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossMEP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMEP, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # Calculating the entropy as the negative sum of P(x) log P(x) over the last dimension
        entropy = -torch.sum(target * torch.log(output + 1e-8), dim=-1).mean()
        mse_loss = self.mse(output, target)
        regularized_loss = self.alpha * mse_loss + (1 - self.alpha) * entropy
        return regularized_loss

class LossMLP(nn.Module):
    def __init__(self, alpha=0.5):
        super(LossMLP, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        # Assuming that the maximum likelihood estimation under a Gaussian assumption reduces to NIL
        likelihood_loss = F.nll_loss(output, target)
        regularized_loss = self.alpha * likelihood_loss
        return regularized_loss