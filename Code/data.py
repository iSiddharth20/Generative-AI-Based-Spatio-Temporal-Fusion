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
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        # Return the training and validation dataloaders
        return train_loader, val_loader

    # Transform a sequence of images to tensors (Functionality for LSTM)
    def transform_sequence(self, filenames):
        print('Executing transform_sequence of CustomDataset Class from data.py')
        images = [self.transform(Image.open(f)) for f in filenames]
        print(f"Transformed sequence shape: {torch.stack(images).shape}")
        return torch.stack(images) # Stack to form a sequence tensor
    
    # Get batches for LSTM training
    def get_lstm_batches(self, val_split, sequence_length, sequence_stride=2):
        print('Executing get_lstm_batches of CustomDataset Class from data.py')
        assert sequence_length % 2 == 0, "The sequence length must be even."
        
        # Compute the total number of sequences that can be formed, given the stride and length
        sequence_indices = range(0, len(self.filenames) - sequence_length + 1, sequence_stride)
        total_sequences = len(sequence_indices)
        
        # Divide the sequences into training and validation
        train_size = int((1.0 - val_split) * total_sequences)
        train_indices = sequence_indices[:train_size]
        val_indices = sequence_indices[train_size:]

        # Create dataset with valid sequences only
        train_dataset = self.create_sequence_pairs(train_indices, sequence_length)
        val_dataset = self.create_sequence_pairs(val_indices, sequence_length)
        
        # Create the data loaders for training and validation datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print("Sample from lstm training data:")
        for input_seq, target_seq in train_loader:
            print(f"Sample input shape in train_loader: {input_seq.shape}")
            print(f"Sample target shape in train_loader: {target_seq.shape}")
            break
        print("Sample from lstm validation data:")
        for input_seq, target_seq in val_loader:
            print(f"Sample input shape in val_loader: {input_seq.shape}")
            print(f"Sample target shape in val_loader: {target_seq.shape}")
            break

        return train_loader, val_loader

    def create_sequence_pairs(self, indices, sequence_length):
        print('Executing create_sequence_pairs of CustomDataset Class from data.py')
        sequence_pairs = []
        for start in indices:
            end = start + sequence_length
            # Make sure we don't go out of bounds
            if end < len(self.filenames):
                sequence_input = self.transform_sequence(self.filenames[start:end])
                sequence_target = self.transform_sequence(self.filenames[start + 1:end + 1])
                sequence_pairs.append((sequence_input, sequence_target))
            else:
                # Handle the last sequence by either discarding or padding
                pass  # Choose to either discard (do nothing) or pad the sequence
        return sequence_pairs