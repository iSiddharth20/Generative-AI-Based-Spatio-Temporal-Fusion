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

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, grayscale_dir, rgb_dir, image_size, batch_size, valid_exts=['.tif', '.tiff']):
        # Initialize directory paths and parameters
        self.grayscale_dir = Path(grayscale_dir)  # Directory for grayscale images
        self.rgb_dir = Path(rgb_dir)  # Directory for RGB images
        self.image_size = image_size  # Size to which images will be resized
        self.batch_size = batch_size  # Batch size for training
        self.valid_exts = valid_exts  # Valid file extensions
        # Get list of valid image filenames
        self.filenames = [f for f in self.grayscale_dir.iterdir() if f.suffix in self.valid_exts]
        # Define transformations: resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])

    # Return the total number of images
    def __len__(self):
        return len(self.filenames)

    # Get a single item or a slice from the dataset
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        # Get paths for grayscale and RGB images
        grayscale_path = self.filenames[idx]
        rgb_path = self.rgb_dir / grayscale_path.name
        # Open images
        grayscale_img = Image.open(grayscale_path)
        rgb_img = Image.open(rgb_path)
        # Apply transformations
        grayscale_img = self.transform(grayscale_img)
        rgb_img = self.transform(rgb_img)
        # Return transformed images
        return grayscale_img, rgb_img

    # Get batches for autoencoder training
    def get_autoencoder_batches(self, val_split):
        # Calculate the number of samples to include in the validation set
        val_size = int(val_split * len(self))
        train_size = len(self) - val_size
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        # Create dataloaders for the training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print("Sample from autoencoder training data:")
        for sample in train_loader:
            print(f'Input shape: {sample[0].shape}, Target shape: {sample[1].shape}')
            break  # Just print the first sample and break
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        # Return the training and validation dataloaders
        return train_loader, val_loader

    # Get batches for LSTM training
    def get_lstm_batches(self, val_split, n=1):
        # Calculate the number of samples to include in the validation set
        val_size = int(val_split * (len(self) // 2))  # Half of sequences because we use every second image.
        train_size = (len(self) // 2) - val_size

        # Get indices for the odd (input) and even (target) frames.
        odd_indices = list(range(0, len(self), 2))
        even_indices = list(range(1, len(self), 2))

        # Split the dataset indices into training and validation subsets
        train_odd_indices = odd_indices[:train_size]
        val_odd_indices = odd_indices[train_size:]

        train_even_indices = even_indices[:train_size]
        val_even_indices = even_indices[train_size:]

        # Define a helper function to extract sequences by indices
        def extract_sequences(indices):
            return [self[i] for i in indices]

        # Use the helper function to create training and validation sets
        train_input_seqs = torch.stack(extract_sequences(train_odd_indices))
        train_target_seqs = torch.stack(extract_sequences(train_even_indices))

        val_input_seqs = torch.stack(extract_sequences(val_odd_indices))
        val_target_seqs = torch.stack(extract_sequences(val_even_indices))

        # Create custom Dataset for the LSTM sequences
        class LSTMDataset(Dataset):
            def __init__(self, input_seqs, target_seqs):
                self.input_seqs = input_seqs
                self.target_seqs = target_seqs

            def __len__(self):
                return len(self.input_seqs)

            def __getitem__(self, idx):
                return self.input_seqs[idx], self.target_seqs[idx]

        # Instantiate the custom Dataset objects
        train_dataset = LSTMDataset(train_input_seqs, train_target_seqs)
        val_dataset = LSTMDataset(val_input_seqs, val_target_seqs)

        # Create DataLoaders for the LSTM datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Return the training and validation DataLoaders
        return train_loader, val_loader

