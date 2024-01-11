'''
Module for AutoEncoder
Generates 3-Chanel RGB Image from 1-Chanel Grayscale Image
--------------------------------------------------------------------------------
For each pair of consecutive values in the channels list, a Convolutional or Transposed Convolutional layer is created.
The number of input channels is the first value, and the number of output channels is the second value.
A Batch Normalization layer and a LeakyReLU activation function are added after each Convolutional or Transposed Convolutional layer.
In the case of the decoder, the final layer uses a Sigmoid activation function instead of LeakyReLU.
'''

# Import Necessary Libraries
import torch.nn as nn

# Define AutoEncoder Architecture
class Grey2RGBAutoEncoder(nn.Module):
    def __init__(self):  
        super(Grey2RGBAutoEncoder, self).__init__()  
        # Define the Encoder
        self.encoder = self._make_layers([3, 4, 8, 16, 32, 64, 128])
        # Define the Decoder
        self.decoder = self._make_layers([128, 64, 32, 16, 8, 4, 3], decoder=True)

    # Helper function to create the encoder or decoder layers.
    def _make_layers(self, channels, decoder=False):
        layers = []
        for i in range(len(channels) - 1):
            if decoder:
                layers += [nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                           nn.ReLU(inplace=True)]
        if decoder:
            layers[-1] = nn.Sigmoid() 
        return nn.Sequential(*layers)

    # The forward pass takes an input image, passes it through the encoder and decoder, and returns the output image
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
