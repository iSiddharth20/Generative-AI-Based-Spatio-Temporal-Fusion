'''
Module that specifies AutoEncoder Architecture for AutoEncoder using PyTorch
--------------------------------------------------------------------------------
This is a Template Code and Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries
import torch.nn as nn

# Define a Class which creates AutoEncoder with Convolutional Layers using PyTorch
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder Architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1), # (N, 16, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # (N, 16, 14, 14)
            nn.Conv2d(16, 8, 3, stride=1, padding=1), # (N, 8, 14, 14)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # (N, 8, 7, 7)
            nn.Conv2d(8, 8, 3, stride=1, padding=1), # (N, 8, 7, 7)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) # (N, 8, 3, 3)
        )
        
        # Decoder Architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2), # (N, 8, 7, 7)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 5, stride=2, padding=1), # (N, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 2, stride=2, padding=1), # (N, 16, 31, 31)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1), # (N, 1, 64, 64)
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x