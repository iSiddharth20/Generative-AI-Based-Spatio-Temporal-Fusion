'''
Module that specifies Encoder Architecture for AutoEncoder using PyTorch
--------------------------------------------------------------------------------
This is a Sample Code and Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries
import torch.nn as nn

# Define a Class called Encoder which creates a simple Encoder of AutoEncoder using PyTorch
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # Initialize the super class
        super(Encoder, self).__init__()

        # Define the input size to the Encoder
        self.input_size = input_size

        # Define the hidden size of the Encoder
        self.hidden_size = hidden_size

        # Define the number of layers of the Encoder
        self.num_layers = num_layers

        # Define the dropout parameter of the Encoder
        self.dropout = dropout
    
    # Define a Function that specifies the Forward Pass of the Encoder
    def forward(self, x):
        # Define the Forward Pass of the Encoder
        return None
