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
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size[2], image_size[3]
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


# Frame Interpolation model using ConvLSTM cells
class FrameInterpolationLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(FrameInterpolationLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_lstm_cells = nn.ModuleList([
            ConvLSTMCell(input_channels=(input_dim if i == 0 else hidden_dim),
                         hidden_channels=hidden_dim,
                         kernel_size=kernel_size,
                         bias=True)
            for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(in_channels=hidden_dim,
                              out_channels=input_dim,
                              kernel_size=(1, 1),
                              padding=0)

    def forward(self, x, n):
        batch_size, seq_len, _, h, w = x.shape
        layer_input = x
        hidden_states = []
        cell_states = []

        for i in range(self.num_layers):
            h, c = self.conv_lstm_cells[i].init_hidden(batch_size, x.shape)
            hidden_states.append(h)
            cell_states.append(c)

        interpolated_sequences = [x[:, :1]]
        for t in range(seq_len - 1):
            current_frame = layer_input[:, t]
            next_frame = layer_input[:, t + 1]
            interpolations = [current_frame]

            for layer in range(self.num_layers):
                h_cur, c_cur = hidden_states[layer], cell_states[layer]
                h_next, c_next = self.conv_lstm_cells[layer](current_frame.unsqueeze(1), (h_cur, c_cur))
                current_frame = h_next[:, -1]

                h_n, c_n = self.conv_lstm_cells[layer](next_frame.unsqueeze(1), (h_cur, c_cur))
                next_frame_pred = h_n[:, -1]
                hidden_states[layer], cell_states[layer] = h_n, c_n

            for j in range(1, n + 1):
                alpha = j / float(n + 1)
                interpolated_frame = (1 - alpha) * current_frame + alpha * next_frame_pred
                interpolated_frame = F.relu(self.conv(interpolated_frame))
                interpolations.append(interpolated_frame)
            interpolations.append(next_frame)

            seq_interpolations = torch.cat(interpolations, dim=1)
            interpolated_sequences.append(seq_interpolations)

        full_sequence = torch.cat(interpolated_sequences, dim=1)
        return full_sequence