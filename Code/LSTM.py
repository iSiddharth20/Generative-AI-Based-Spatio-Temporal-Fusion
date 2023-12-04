'''
Module that specifies LSTM Network using PyTorch
--------------------------------------------------------------------------------
Create a simple LSTM network using PyTorch, assume that CUDA GPU is available. 
Take any random size for input and hidden size. 
'''

# Importing Necessary Libraries
import torch
import torch.nn as nn

# Defining the LSTM Network
class LSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModule, self).__init__()
        # Define the parameters for the LSTM layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM layer
        # x of shape (batch, seq_length, input_size): tensor containing the features of the input sequence.
        # h0 of shape (num_layers, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        # c0 of shape (num_layers, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        # output of shape (batch, seq_length, hidden_size): tensor containing the output features from the last layer of the LSTM, for each t.
        # hn, cn: tensors containing the hidden and cell state of the last element of the sequence.
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # The LSTM output can be used to make predictions if needed
        predictions = self.linear(output[:, -1, :]) # Consider the last output for prediction
        return predictions
    
