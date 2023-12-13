'''
Module that specifies LSTM Network using PyTorch
--------------------------------------------------------------------------------
'''

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size, sequence_length, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.output_size = output_size
        
        # Define an LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_features, 
            hidden_size=self.hidden_size, 
            batch_first=True
        )
        
        # Define a dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Define a fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Define a fully connected layer for the interpolation weights
        self.fc_interp = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Reshape the data for the Fully Connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        
        # Pass the output of the last time step to the FC layer
        out = self.fc(out)
        
        # Compute the interpolation weights
        interp_weights = self.fc_interp(out)
        
        return out, interp_weights