'''
Module that specifies Data Pre-Processing
--------------------------------------------------------------------------------
'''

import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

class Dataset:
    def __init__(self, grayscale_dir='../Dataset/Greyscale', rgb_dir='../Dataset/RGB', image_size=(400, 600), batch_size=64, augment=False):
        self.grayscale_dir = grayscale_dir
        self.rgb_dir = rgb_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        # Load images separately from their respective directories
        self.grayscale_images = self.load_images(self.grayscale_dir, convert_to_grayscale=True)
        self.rgb_images = self.load_images(self.rgb_dir, convert_to_grayscale=True)
    
    def load_images(self, directory, convert_to_grayscale):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                if convert_to_grayscale:
                    img = img.convert('L')  # Convert to grayscale if necessary
                img = img.resize(self.image_size)
                if self.augment:
                    img = self.augment_image(img)
                images.append(np.array(img))
        return images
    
    def preprocess_image(self, img):
        # Resize image
        img = img.resize(self.image_size)
        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')
        # Apply augmentations if enabled
        if self.augment:
            img = self.augment_image(img)
        return img

    def augment_image(self, img):
        # Define augmentations
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            # Add any other transformations you want here
        ])
        img = augmentations(img)
        return img

    def normalize_images(self, images):
        # Convert list to numpy array, then normalize images to be in the range [0, 1]
        return np.array(images).astype('float32') / 255.0

    def split_data(self):
        # Split the grayscale images into train/val/test
        train_gray, test_gray = train_test_split(self.grayscale_images, test_size=0.2, random_state=42)
        train_gray, val_gray = train_test_split(train_gray, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        # Split the RGB images into train/val/test
        train_rgb, test_rgb = train_test_split(self.rgb_images, test_size=0.2, random_state=42)
        train_rgb, val_rgb = train_test_split(train_rgb, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        # Normalize images
        train_gray, val_gray, test_gray = map(self.normalize_images, [train_gray, val_gray, test_gray])
        train_rgb, val_rgb, test_rgb = map(self.normalize_images, [train_rgb, val_rgb, test_rgb])

        return (train_gray, val_gray, test_gray), (train_rgb, val_rgb, test_rgb)

    def get_batches(self, data):
        # Generate batches
        data = torch.tensor(data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader