'''
Module for LSTM
Generate Intermediate Images and Return the Complete Image Sequence with Interpolated Images
--------------------------------------------------------------------------------
'''
# Import Necessary Libraries
import torch
from torch import nn
from torch.nn import functional as F

class ConvLSTMCell(nn.Module):     
    def __init__(self, input_dim, hidden_dim, kernel_size, num_features):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=num_features * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, alpha=0.5):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cells.append(ConvLSTMCell(input_dim=cur_input_dim,
                                           hidden_dim=hidden_dims[i],
                                           kernel_size=kernel_size,
                                           num_features=4))  # LSTM has 4 gates (features)

    def init_hidden(self, batch_size, image_height, image_width):
        init_states = []
        for i in range(self.num_layers):
            # Note the change from self.hidden_dim to self.hidden_dims
            init_states.append([torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device),
                                torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device)])
        return init_states
    
  
    def forward(self, input_tensor, cur_state=None):
        b, seq_len, _, h, w = input_tensor.size()

        if cur_state is None:
            cur_state = self.init_hidden(b, h, w)

        # Initialize output tensors for each sequence element
        output_sequence = torch.zeros((b, seq_len - 1, self.hidden_dims[-1], h, w), device=input_tensor.device)

        for layer_idx, cell in enumerate(self.cells):
            
            # Fix: Unpack hidden and cell states for the current layer
            h, c = cur_state[layer_idx]
            
            # For handling the sequence of images
            for t in range(seq_len - 1):
                # Perform forward pass through the cell
                h, c = cell(input_tensor[:, t, :, :, :], (h, c)) # Updated to pass tuple `(h, c)`

                if layer_idx == self.num_layers - 1: # Only store output from the last layer
                    output_sequence[:, t, :, :, :] = h
                    
                # Generate the next input from alpha-blending
                if t != seq_len - 2:
                    next_input = (1 - self.alpha) * input_tensor[:, t, :, :, :] + self.alpha * input_tensor[:, t + 1, :, :, :]
                    h, c = cell(next_input, (h, c)) # Updated to pass tuple `(h, c)`

            cur_state[layer_idx] = (h, c)
            
        # No need to stack since we're assigning the results in the output tensor

        # Predict an extra frame beyond the last input frame
        h, c = cell(input_tensor[:, -1, :, :, :], (h, c))
        output_sequence = torch.cat([output_sequence, h.unsqueeze(1)], dim=1)
               
        return output_sequence, cur_state