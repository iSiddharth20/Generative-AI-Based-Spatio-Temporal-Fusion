'''
Module that specifies Data Pre-Processing
Importing Dataset, Converting it to PyTorch Tensors, Splitting it into Training and Validation Sets
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import DataLoader, TensorDataset

class Dataset:
    def __init__(self, grayscale_dir, rgb_dir, image_size, batch_size):
        self.grayscale_dir = grayscale_dir  # Directory for grayscale images
        self.rgb_dir = rgb_dir  # Directory for RGB images
        self.image_size = image_size  # Size to which images will be resized
        self.batch_size = batch_size  # Batch size for training
        '''
        Load Greyscale and RGB images from respective directories
        Store All Images of the Directory in a Normalized NumPy arrays
        Convert the NumPy arrays to PyTorch Tensors
        '''
        self.grayscale_images = self.load_images_to_tensor(self.grayscale_dir)
        self.rgb_images = self.load_images_to_tensor(self.rgb_dir)

    # Function to load images, resize and export as NumPy array
    def load_images_to_tensor(self, directory):
        images = []
        # Loop through all files in the directory
        for filename in os.listdir(directory):
            # If the file is an image file
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                # Resize the image
                img = img.resize(self.image_size)
                # Append the normalized image to the list
                img_array = np.array(img).astype('float32') / 255.0
                # Add an extra dimension for grayscale images
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                images.append(img_array)
        # Return the PyTorch Tensor {with shape [m, C, H, W]} created from of NumPy Array of Images
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        return images
    
    # Function to get batches of input-target pairs from data (This Functionality is for AutoEncoder Component of the Program)
    def get_autoencoder_batches(self,val_split):
        # Create a Dataset from the Tensors
        dataset = TensorDataset(self.grayscale_images, self.rgb_images)
        # Calculate the number of samples to include in the validation set
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        # Create dataloaders for the training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        # Return the training and validation dataloaders
        return train_loader, val_loader

    # Function to get batches of original_sequence-interpolated_sequence from data (This Functionality is for LSTM Component of the Program)
    def get_lstm_batches(self, val_split):
        # Assumption: self.grayscale_images is a tensor with shape [num_samples, channels, height, width]
        
        # Calculate the number of frames for training and validation
        num_frames = self.grayscale_images.size(0)
        train_frames = int(num_frames * (1 - val_split))
        val_frames = num_frames - train_frames
        
        # Split the frames into training and validation sets
        train_frames_tensor = self.grayscale_images[:train_frames]
        val_frames_tensor = self.grayscale_images[-val_frames:]
        
        # Create sequences (8 frames per sequence here) for training and validation
        # The goal is to predict 7 intermediate frames from 8 input frames
        input_seq_len = 8
        target_seq_len = 7
        
        train_seqs = train_frames_tensor.unfold(0, input_seq_len, 1)
        train_targets = train_seqs.roll(-1, dims=0)[:, :target_seq_len]  # Shift temporal sequence by one step
        
        val_seqs = val_frames_tensor.unfold(0, input_seq_len, 1)
        val_targets = val_seqs.roll(-1, dims=0)[:, :target_seq_len]  # Shift temporal sequence by one step
        
        # Make sure there is no overlap between training and validation targets
        if train_seqs[-1].equals(val_seqs[0]):
            val_seqs = val_seqs[1:]
            val_targets = val_targets[1:]
        
        # Create tensor datasets and data loaders
        train_dataset = TensorDataset(train_seqs, train_targets)
        val_dataset = TensorDataset(val_seqs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
