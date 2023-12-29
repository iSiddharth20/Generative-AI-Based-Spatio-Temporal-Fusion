'''
Module for LSTM
Generate Intermediate Images and Return the Complete Image Sequence with Interpolated Images
--------------------------------------------------------------------------------
'''
# Importing Necessary Libraries
import torch
from torch import nn
from torch.nn import functional as F

# Define ConvLSTMCell class
class ConvLSTMCell(nn.Module):     
    def __init__(self, input_dim, hidden_dim, kernel_size, num_features):
        super(ConvLSTMCell, self).__init__()
        # Define the convolutional layer
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=num_features * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        # Unpack the current state into hidden state (h_cur) and cell state (c_cur)
        h_cur, c_cur = cur_state
        # Concatenate the input tensor and the current hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        # Apply the convolution to the combined tensor
        combined_conv = self.conv(combined)
        # Split the convolution output into four parts for input gate, forget gate, output gate, and cell gate
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # Apply sigmoid activation to the input, forget, and output gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        # Apply tanh activation to the cell gate
        g = torch.tanh(cc_g)
        # Compute the next cell state as a combination of the forget gate, current cell state, input gate, and cell gate
        c_next = f * c_cur + i * g
        # Compute the next hidden state as the output gate times the tanh of the next cell state
        h_next = o * torch.tanh(c_next)
        # Return the next hidden state and cell state
        return h_next, c_next

# Define the ConvLSTM class
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, alpha=0.5):
        super(ConvLSTM, self).__init__()
        # Set the number of layers, alpha parameter, and hidden dimensions
        self.num_layers = num_layers
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        # Initialize a ModuleList to hold the ConvLSTM cells
        self.cells = nn.ModuleList()
        # Loop over the number of layers and create a ConvLSTM cell for each layer
        for i in range(num_layers):
            # The input dimension for the first layer is input_dim, for other layers it is the hidden dimension of the previous layer
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            # Append a new ConvLSTM cell to the cells list
            self.cells.append(ConvLSTMCell(input_dim=cur_input_dim,
                                           hidden_dim=hidden_dims[i],
                                           kernel_size=kernel_size,
                                           num_features=4))  # LSTM has 4 gates (features)

    def init_hidden(self, batch_size, image_height, image_width):
        # Initialize a list to hold the initial hidden and cell states
        init_states = []
        # Loop over the number of layers
        for i in range(self.num_layers):
            '''
            For each layer, create a zero tensor for the hidden state and the cell state
            The size of the tensor is (batch_size, hidden_dim, image_height, image_width)
            The tensor is moved to the same device as the weights of the convolutional layer of the corresponding ConvLSTM cell
            '''
            init_states.append([torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device),
                                torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device)])
        # Return the initial states
        return init_states
    
    def forward(self, input_tensor, cur_state=None):
        # Extract the batch size, sequence length, height, and width from the input tensor
        b, seq_len, _, h, w = input_tensor.size()
        # If no current state is provided, initialize it using the init_hidden method
        if cur_state is None:
            cur_state = self.init_hidden(b, h, w)
        # Initialize the output sequence tensor with zeros
        output_sequence = torch.zeros((b, seq_len - 1, self.hidden_dims[-1], h, w), device=input_tensor.device)
        # Loop over each ConvLSTM cell (layer) in the model
        for layer_idx, cell in enumerate(self.cells):
            # Extract the hidden state and cell state for the current layer
            h, c = cur_state[layer_idx]
            # Loop over each time step in the input sequence
            for t in range(seq_len - 1):
                # Pass the input and current state through the cell to get the next state
                h, c = cell(input_tensor[:, t, :, :, :], (h, c))
                # If this is the last layer, add the hidden state to the output sequence
                if layer_idx == self.num_layers - 1:
                    output_sequence[:, t, :, :, :] = h
                # If this is not the last time step, generate the next input by alpha-blending the current and next input
                if t != seq_len - 2:
                    next_input = (1 - self.alpha) * input_tensor[:, t, :, :, :] + self.alpha * input_tensor[:, t + 1, :, :, :]
                    h, c = cell(next_input, (h, c))
            # Update the current state for this layer
            cur_state[layer_idx] = (h, c)
        # After processing all time steps, predict an extra frame beyond the last input frame
        h, c = cell(input_tensor[:, -1, :, :, :], (h, c))
        output_sequence = torch.cat([output_sequence, h.unsqueeze(1)], dim=1)
        # Return the output sequence and the final state
        return output_sequence, cur_state
