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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        '''
        # Define the Decoder
        The Decoder consists of 4 Transpose Convolutional layers with ReLU activation function
        Decoder takes High-Dimentional-Representation as input and outputs 3-Chanel RGB image
        '''
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    # The forward pass takes an input image, passes it through the encoder and decoder, and returns the output image
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

