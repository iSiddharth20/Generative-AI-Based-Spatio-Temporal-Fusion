'''
Module for LSTM
Generate Intermediate Frames and Return the Interpolated Image Sequence
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional LSTM cell for processing sequences of images.
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        # Store the number of input channels, hidden channels, kernel size, and bias
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.padding = kernel_size // 2
        elif isinstance(kernel_size, tuple):
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            raise ValueError("kernel_size must be an int or a tuple of two ints")
        self.bias = bias
        # Define a convolutional layer that takes both the input and the hidden state as input
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,  # Four sets of filters for the LSTM gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    def forward(self, input_tensor, cur_state):
        # Unpack the current state
        h_cur, c_cur = cur_state
        # Concatenate the input tensor and the current hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        # Apply the convolutional layer
        combined_conv = self.conv(combined)
        # Split the convolutional output into four parts for the LSTM gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        # Compute the values for the input, forget, output gates and cell state
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # Compute the next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        # Return the next hidden state and cell state
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        # Initialize the hidden state and cell state to zeros
        # height, width = image_size
        height, width = int(image_size[0]), int(image_size[1])
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))
    
# Frame Interpolation model using ConvLSTM cells
class FrameInterpolationLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(FrameInterpolationLSTM, self).__init__()
        # Store the hidden dimension and number of layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Define a list of ConvLSTM cells
        self.conv_lstm_cells = nn.ModuleList([
            ConvLSTMCell(input_channels=(input_dim if i == 0 else hidden_dim),
                         hidden_channels=hidden_dim,
                         kernel_size=kernel_size,
                         bias=True)
            for i in range(num_layers)
        ])
        # Define a pointwise convolutional layer to map the LSTM output to the desired output feature space
        self.conv = nn.Conv2d(in_channels=hidden_dim,
                              out_channels=input_dim,  # Assuming we want to generate grayscale images
                              kernel_size=(1, 1),  # pointwise convolution
                              padding=0)
    def forward(self, x, n):
        # Get the batch size, sequence length, and image height and width from the input
        batch_size, seq_len, _, h, w = x.shape
        # Initialize the input for the first layer and the lists of hidden and cell states
        layer_input = x
        hidden_states = []
        cell_states = []
        # Initialize the hidden and cell states for each layer
        for i in range(self.num_layers):
            h, c = self.conv_lstm_cells[i].init_hidden(batch_size, (h, w))
            hidden_states.append(h)
            cell_states.append(c)
        # Initialize the list of interpolated sequences with the first frame
        interpolated_sequences = [x[:, :1]]
        # Loop over each time step
        for t in range(seq_len - 1):
            # Get the current frame and the next frame
            current_frame = layer_input[:, t]
            next_frame = layer_input[:, t + 1]
            interpolations = [current_frame]
            # Loop over each layer
            for layer in range(self.num_layers):
                # Update the current frame and the hidden and cell states
                hidden_states[layer], cell_states[layer] = self.conv_lstm_cells[layer](current_frame, (hidden_states[layer], cell_states[layer]))
                current_frame = hidden_states[layer]
                # Predict the next frame and store the next hidden and cell states
                next_frame_pred, (h_n, c_n) = self.conv_lstm_cells[layer](next_frame, (hidden_states[layer], cell_states[layer]))
                hidden_states_next, cell_states_next = h_n, c_n
            # Generate `n` interpolated frames between the current frame and the predicted next frame
            for j in range(1, n + 1):
                alpha = j / float(n + 1)
                interpolated_frame = (1 - alpha) * current_frame + alpha * next_frame_pred
                interpolated_frame = F.relu(self.conv(interpolated_frame))
                interpolations.append(interpolated_frame)
            # Append the true next frame and update the hidden and cell states for the next true frame
            interpolations.append(next_frame)
            for layer in range(self.num_layers):
                hidden_states[layer], cell_states[layer] = hidden_states_next[layer], cell_states_next[layer]
            # Concatenate the interpolations along the time dimension and append them to the list of interpolated sequences
            seq_interpolations = torch.cat(interpolations, dim=1)
            interpolated_sequences.append(seq_interpolations)
        # Concatenate all sequences (original and interpolated) along the time dimension
        full_sequence = torch.cat(interpolated_sequences, dim=1)
        # Return the full sequence
        return full_sequence
    
