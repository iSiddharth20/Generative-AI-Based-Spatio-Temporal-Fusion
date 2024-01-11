'''
Module that specifies Data Pre-Processing
Importing Dataset, Converting it to PyTorch Tensors, Splitting it into Training and Validation Sets
--------------------------------------------------------------------------------
'''

# Import Necessary Libraries
from PIL import Image
from PIL import ImageFile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torch
from torch.utils.data.distributed import DistributedSampler

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, autoencoder_grayscale_dir, autoencoder_rgb_dir, lstm_gray_sequences_dir, image_size, valid_exts=['.tif', '.tiff']):
        # Initialize directory paths and parameters
        self.grayscale_dir = Path(autoencoder_grayscale_dir)  # Directory for AutoEncoder grayscale images
        self.rgb_dir = Path(autoencoder_rgb_dir)  # Directory for AutoEncoder RGB images
        self.lstm_dir = Path(lstm_gray_sequences_dir)  # Directory for LSTM grayscale sequences
        self.image_size = image_size  # Size to which images will be resized
        self.valid_exts = valid_exts  # Valid file extensions
        # Get list of valid image filenames
        self.autoencoder_filenames = [f for f in self.grayscale_dir.iterdir() if f.suffix in self.valid_exts]
        self.lstm_filenames = [f for f in self.lstm_dir.iterdir() if f.suffix in self.valid_exts]
        self.autoencoder_length = len(self.autoencoder_filenames)
        self.lstm_length = len(self.lstm_filenames)
        # Define transformations: resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])

    # Return the total number of images
    def __len__(self, lstm=False):
        return self.lstm_length if lstm else self.autoencoder_length

    # Get a single item or a slice from the dataset
    def __getitem__(self, idx, lstm=False):
        # Get paths for grayscale and RGB images
        grayscale_path = self.lstm_filenames[idx] if lstm else self.autoencoder_filenames[idx]
        rgb_path = self.rgb_dir / grayscale_path.name
        # Open images
        try:
            grayscale_img = Image.open(grayscale_path)
            rgb_img = Image.open(rgb_path) if not lstm else None
        except IOError:
            print(f"Error opening images {grayscale_path} or {rgb_path}")
            return None
        # Apply transformations
        grayscale_img = self.transform(grayscale_img)
        if rgb_img is not None:
            rgb_img = self.transform(rgb_img)
        # Return transformed images
        return grayscale_img, rgb_img

    # Get batches for autoencoder training
    def get_autoencoder_batches(self, val_split, batch_size):
        # Calculate the number of samples to include in the validation set
        val_size = int(val_split * len(self))
        train_size = len(self) - val_size
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        # Create dataloaders for the training and validation sets
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, sampler=val_sampler)
        # Return the training and validation dataloaders
        return train_loader, val_loader

    # Get batches for LSTM training
    def get_lstm_batches(self, val_split, sequence_length, batch_size):
        assert sequence_length % 2 == 0, "The sequence length must be even."
        # Create a list of all possible start indices for the sequences
        sequence_indices = list(range(0, self.__len__(lstm=True), sequence_length))
        # Create dataset with valid sequences only
        train_dataset = []
        val_dataset = []
        train_end = int((1.0 - val_split) * sequence_length)
        for start in sequence_indices:
            end = start + sequence_length
            # Make sure we don't go out of bounds
            if end <= self.__len__(lstm=True):
                sequence = self.transform_sequence(self.lstm_filenames[start:end], lstm=True)
                sequence_input_train = sequence[:train_end:2]  # Odd-indexed images for training
                sequence_target_train = sequence[1:train_end:2]  # Even-indexed images for training
                sequence_input_val = sequence[train_end::2]  # Odd-indexed images for validation
                sequence_target_val = sequence[train_end+1::2]  # Even-indexed images for validation
                train_dataset.append((sequence_input_train, sequence_target_train))
                val_dataset.append((sequence_input_val, sequence_target_val))
        # Create the data loaders for training and validation datasets
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, sampler=val_sampler)
        return train_loader, val_loader

    def transform_sequence(self, filenames, lstm=False):
        images = [self.transform(Image.open(f if not lstm else self.lstm_dir / f.name)) for f in filenames]
        return torch.stack(images) # Stack to form a sequence tensor