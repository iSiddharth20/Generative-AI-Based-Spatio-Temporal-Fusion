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
from torch.utils.data import DataLoader, TensorDataset, Dataset


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
        greyscale_image_sequence = self.grayscale_images.unsqueeze(0)
        num_frames = greyscale_image_sequence.size(1)
        split_idx = int(num_frames * val_split) 
        # Ensure even number of frames for splitting into pairs.
        split_idx = split_idx if split_idx % 2 == 0 else split_idx - 1
        greyscale_image_sequence_train = greyscale_image_sequence[:, :split_idx, :, :, :]
        greyscale_image_sequence_val = greyscale_image_sequence[:, split_idx:, :, :, :]
        # Ensure the same length for both odd and even splits.
        even_indices = range(0, split_idx, 2)
        odd_indices = range(1, split_idx, 2)
        train_data = TensorDataset(greyscale_image_sequence_train[:, even_indices].squeeze(dim=0), 
                                    greyscale_image_sequence_train[:, odd_indices].squeeze(dim=0))
        even_indices = range(0, greyscale_image_sequence_val.size(1), 2)
        odd_indices = range(1, greyscale_image_sequence_val.size(1), 2)
        val_data = TensorDataset(greyscale_image_sequence_val[:, even_indices].squeeze(dim=0),
                                    greyscale_image_sequence_val[:, odd_indices].squeeze(dim=0))
        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader
