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
    def get_lstm_batches(self, val_split):
        # Calculate the split index
        num_frames = len(self)
        split_idx = int(num_frames * val_split) 
        # Ensure even number of frames for splitting into pairs
        split_idx = split_idx if split_idx % 2 == 0 else split_idx - 1
        # Split grayscale images into training and validation sets
        grayscale_images_train = self[:split_idx]
        grayscale_images_val = self[split_idx:]
        # Ensure the same length for both odd and even splits
        odd_indices = range(1, split_idx, 2)
        even_indices = range(0, split_idx, 2)
        # Create training and validation data
        train_data = [grayscale_images_train[i] for i in odd_indices]
        val_data = [grayscale_images_val[i] for i in even_indices]
        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        print("Sample from LSTM training data:")
        for sample in train_loader:
            print(f'Input sequence shape: {sample[0].shape}, Target sequence shape: {sample[1].shape}')
            break  # Just print the first sample and break
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        # Return the training and validation dataloaders
        return train_loader, val_loader

