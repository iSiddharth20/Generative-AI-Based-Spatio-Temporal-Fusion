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
        print('Executing __init__ of ConvLSTMCell Class from lstm_model.py')
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=num_features * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, input_tensor, cur_state):
        print('Executing forward of ConvLSTMCell Class from lstm_model.py')
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        print(f'ConvLSTMCell combined input shape (before conv): {combined.shape}')
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
        print('Executing __init__ of ConvLSTM Class from lstm_model.py')
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
        print('Executing init_hidden of ConvLSTM Class from lstm_model.py')
        init_states = []
        for i in range(self.num_layers):
            # Note the change from self.hidden_dim to self.hidden_dims
            init_states.append([torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device),
                                torch.zeros(batch_size, self.hidden_dims[i], image_height, image_width, device=self.cells[i].conv.weight.device)])
        return init_states
    
    def forward(self, input_tensor, cur_state=None):
        print('Executing forward of ConvLSTM Class from lstm_model.py')
        print(f"Overall Input Tensor Shape: {input_tensor.shape}")
        b, seq_len, _, h, w = input_tensor.size()

        if cur_state is None:
            cur_state = self.init_hidden(b, h, w)
            for h, c in cur_state:
                print(f"Hidden State h Shape: {h.shape}")
                print(f"Hidden State c Shape: {c.shape}")

        output_sequence = []
        last_state_list = []

        for layer_idx, cell in enumerate(self.cells):
            h, c = cur_state[layer_idx]
            internal_output_sequence = []
            for t in range(seq_len - 1):
                print(f"Input tensor shape before LSTM cell: {input_tensor[:, t, :, :, :].shape}")
                h, c = cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h, c])
                print(f"Output tensor shape after LSTM cell: {h.shape}")
                internal_output_sequence.append(h)

                # Interpolation (alpha-blending) between frames for intermediate frame generation
                if t != seq_len - 2:
                    next_input = (1 - self.alpha) * input_tensor[:, t, :, :, :] + self.alpha * input_tensor[:, t + 1, :, :, :]
                    print(f'Alpha-blending input shape (before LSTM cell): {next_input.shape}')
                    h, c = cell(input_tensor=next_input, cur_state=[h, c])
                    print(f'Next hidden state shape (h): {h.shape}')
                    print(f'Next cell state shape (c): {c.shape}')
                    internal_output_sequence.append(h)

            cur_state[layer_idx] = (h, c)
            output_sequence.append(internal_output_sequence[-1])

            # Concatenate the output from each time step to form a sequence
            output_sequence[layer_idx] = torch.stack(internal_output_sequence, dim=1)

        # We take the output from the last layer as the predicted output
        predicted_sequence = output_sequence[-1]

        return predicted_sequence, cur_state