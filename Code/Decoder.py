'''
Module that specifies Decoder Architecture for AutoEncoder using PyTorch
--------------------------------------------------------------------------------
Data Formats Used:
    - low_res_image: [M x N] NumPy Array
    - high_res_image: [M' x N'] NumPy Array
--------------------------------------------------------------------------------
This Code Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries
import torch.nn as nn
from MaxEnt import MaxEnt

# Define a Class called Decoder which creates a simple Decoder of AutoEncoder using PyTorch
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # Initialize the super class
        super(Decoder, self).__init__()

        # Define the input size to the Decoder
        self.input_size = input_size

        # Define the hidden size of the Decoder
        self.hidden_size = hidden_size

        # Define the number of layers of the Decoder
        self.num_layers = num_layers

        # Define the dropout parameter of the Decoder
        self.dropout = dropout

        # Initialize the MaxEnt model
        self.maxent = None

    # Define a Function that specifies the Forward Pass of the Decoder
    def forward(self, x):
        # Use the MaxEnt model to generate the output
        if self.maxent is not None:
            output = self.maxent.MaxEnt()
            return output
        else:
            # Define the Forward Pass of the Decoder
            return None

    # Initialize the MaxEnt model with the given low-resolution and high-resolution images
    def set_maxent(self, low_res_image, high_res_image):
        self.maxent = MaxEnt(low_res_image, high_res_image)