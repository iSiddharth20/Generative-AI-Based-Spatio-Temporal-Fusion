'''
Module that specifies Decoder Architecture with MaxEnt for AutoEncoder using PyTorch
--------------------------------------------------------------------------------
This is a Sample Code and Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries and Modules
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

    '''
        Need to Combine MaxEnt and Decoder
    '''     
    def MaxEntDecoder():
        return None
    
    # Define a Function that specifies the Forward Pass of the Decoder
    def forward(self, x):
        # Define the Forward Pass of the Decoder
        return None
