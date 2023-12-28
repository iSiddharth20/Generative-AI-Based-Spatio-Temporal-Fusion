'''
Module for LSTM
Generate Intermediate Images and Return the Complete Image Sequence with Interpolated Images
--------------------------------------------------------------------------------
'''
import torch
from torch import nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

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
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cur_hidden_dim = self.hidden_dim[i]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=cur_hidden_dim,
                                          kernel_size=self.kernel_size))

        self.cell_list = nn.ModuleList(cell_list)
    def forward(self, input_tensor, cur_state=None, n=1):
        b, seq_len, _, h, w = input_tensor.size()

        if cur_state is None:
            cur_state = self.init_hidden(b, h, w)

        layer_output_list = []
        last_state_list   = []

        for t in range(seq_len - 1):
            x = input_tensor[:, t, :, :, :]
            cur_layer_input = x

            for i in range(self.num_layers):
                h, c = cur_state[i]
                h, c = self.cell_list[i](input_tensor=cur_layer_input, cur_state=[h, c])
                cur_layer_input = h

                if t == seq_len - 2:
                    last_state_list.append([h, c])

            layer_output_list.append(cur_layer_input)

            # Generate n intermediate frames
            for j in range(1, n + 1):
                alpha = j / float(n + 1)
                x = (1 - alpha) * input_tensor[:, t, :, :, :] + alpha * input_tensor[:, t + 1, :, :, :]
                cur_layer_input = x

                for i in range(self.num_layers):
                    h, c = cur_state[i]
                    h, c = self.cell_list[i](input_tensor=cur_layer_input, cur_state=[h, c])
                    cur_layer_input = h

                    if t == seq_len - 2 and j == n:
                        last_state_list.append([h, c])

                layer_output_list.append(cur_layer_input)

        return layer_output_list, last_state_list

    def init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([torch.zeros(b, self.hidden_dim, h, w).to(self.cell_list[i].conv.weight.device),
                                torch.zeros(b, self.hidden_dim, h, w).to(self.cell_list[i].conv.weight.device)])
        return init_states