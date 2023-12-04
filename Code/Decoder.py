'''
Module that specifies Decoder Architecture for AutoEncoder using PyTorch
--------------------------------------------------------------------------------
This Code Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries
import torch.nn as nn

# Define a Class called Decoder which creates a simple Decoder of AutoEncoder using PyTorch
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # Initialize the super class
        super(Decoder, self).__init__()

        # Define the input size to the Encoder
        self.input_size = input_size

        # Define the hidden size of the Encoder
        self.hidden_size = hidden_size

        # Define the number of layers of the Encoder
        self.num_layers = num_layers

        # Define the dropout parameter of the Encoder
        self.dropout = dropout

        # Define ReLU Activation Function
        self.relu = nn.ReLU()
    
    # Define a Function that specifies the Forward Pass of the Encoder
    def forward(self, x):
        # Define the Forward Pass of the Encoder
        self.layer1 = nn.conv3d(in_channels=3, out_channels=3, kernel_size=64, stride=1, padding=1)
        self.layer2 = nn.conv3d(in_channels=3, out_channels=3, kernel_size=16, stride=1, padding=1)
        self.layer3 = nn.conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x