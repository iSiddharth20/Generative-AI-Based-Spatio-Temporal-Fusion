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
        val_size = int(val_split * len(self))
        train_size = len(self) - val_size

        # Split the dataset into training and validation sets
        all_indices = list(range(len(self)))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        # Function to generate input-target sequence pairs
        def generate_sequences(indices):
            input_sequences = []  # To store input sequences (odd-indexed frames)
            target_sequences = []  # To store target sequences (full frame sequences)

            for start_idx in range(0, len(indices) - 1, 2):  # Step by 2 to get odd-indexed frames
                end_idx = start_idx + 2 * n + 1  # Calculate end index of the sequence
                if end_idx > len(indices):
                    break  # If the end index goes beyond the dataset size, stop adding sequences
                
                # Extract the input sequence (odd-indexed frames)
                input_seq_indices = indices[start_idx:end_idx:2]  # Every second frame (odd)
                input_seq = [self[i][0] for i in input_seq_indices]  # Select grayscale image only
                input_sequences.append(torch.stack(input_seq))
                
                # Extract the target sequence (full frames, including intermediate frames)
                target_seq_indices = indices[start_idx:end_idx]  # All frames in the range
                target_seq = [self[i][0] for i in target_seq_indices]  # Select grayscale image only
                target_sequences.append(torch.stack(target_seq))

            return torch.stack(input_sequences), torch.stack(target_sequences)

        # Generate training and validation sequences
        train_input_seqs, train_target_seqs = generate_sequences(train_indices)
        val_input_seqs, val_target_seqs = generate_sequences(val_indices)

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

