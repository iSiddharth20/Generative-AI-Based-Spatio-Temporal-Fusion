'''
Module for LSTM
Generate Intermediate Frames and Return the Interpolated Image Sequence
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(4 * self.hidden_dim)
        )

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

    # def init_hidden(self, batch_size, image_size):
    #     height, width = image_size[0], image_size[1]
    #     return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device),
    #             torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device))
    def init_hidden(self, batch_size, image_size):
        height, width = image_size  # Unpack the image size tuple
        # Initialize hidden and cell states with non-negative dimensions
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            layers.append(ConvLSTMCell(input_dim if i == 0 else hidden_dim,
                                       hidden_dim, kernel_size))
        
        self.conv_lstm_layers = nn.ModuleList(layers)

    def forward(self, input_tensor):
        # h, c = self.init_hidden(input_tensor.size(0), (-2, -1))  # Get the last two dimensions
        h, c = self.init_hidden(input_tensor.size(0), (input_tensor.size(2), input_tensor.size(3)))

        internal_state = []
        outputs = []
        for timestep in range(input_tensor.size(1)):
            x = input_tensor[:, timestep, :, :, :]
            for layer in range(self.num_layers):
                lstm_cell = self.conv_lstm_layers[layer]
                h, c = lstm_cell(x, (h, c))
                internal_state.append((h, c))
            
            outputs.append(h)
            internal_state = []  # Reset the state for the next timestep

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.conv_lstm_layers[i].init_hidden(batch_size, image_size))
        return tuple(init_states)
