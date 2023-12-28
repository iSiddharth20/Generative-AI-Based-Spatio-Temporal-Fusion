'''
Module for AutoEncoder
Generates 3-Chanel RGB Image from 1-Chanel Grayscale Image
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
import torch.nn as nn

# Define AutoEncoder Architecture
class Grey2RGBAutoEncoder(nn.Module):
    def __init__(self):  
        super(Grey2RGBAutoEncoder, self).__init__()  
        '''
        # Define the Encoder
        The Encoder consists of 4 Convolutional layers with ReLU activation function
        Encoder takes 1-Chanel Grayscale image (1 channel) as input and outputs High-Dimentional-Representation
        '''
        self.encoder = self._make_layers([1, 64, 128, 256, 512])

        '''
        # Define the Decoder
        The Decoder consists of 4 Transpose Convolutional layers with ReLU activation function
        Decoder takes High-Dimentional-Representation as input and outputs 3-Chanel RGB image
        The last layer uses a Sigmoid activation function instead of ReLU
        '''
        self.decoder = self._make_layers([512, 256, 128, 64, 3], decoder=True)

    # Helper function to create the encoder or decoder layers.
    def _make_layers(self, channels, decoder=False):
        print('Executing _make_layers from autoencoder_model.py')
        layers = []
        for i in range(len(channels) - 1):
            '''
            For each pair of consecutive values in the channels list, a Convolutional or Transposed Convolutional layer is created.
            The number of input channels is the first value, and the number of output channels is the second value.
            A ReLU activation function is added after each Convolutional layer.
            '''
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
        print('Executing forward from autoencoder_model.py')
        x = self.encoder(x)
        x = self.decoder(x)
        return x

