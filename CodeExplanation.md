# GenAI-Powered Spatio-Temporal Fusion for Video Super-Resolution

#### Based on PyTorch, Install [Here](https://pytorch.org/get-started/locally/)
#### Access Complete Dataset used in the Study : [Here](https://www.kaggle.com/datasets/isiddharth/spatio-temporal-data-of-moon-rise-in-raw-and-tif)
#### Access Latest Updates [Here](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion)

## Introduction
Developing a novel approach to video super-resolution by harnessing the potential of Auto-Encoders, LSTM Networks, and the Maximum Entropy Principle. The project aims to refine the spatial and temporal resolution of video data, unlocking new possibilities in high-resolution, high-fps, more-color-dense videos and beyond.

## Research Objective
The main goals of the project are:
- To learn temporal dependencies among spatially-sparse-temporally-dense greyscale image frames to predict and interpolate new frames, hence, increasing temporal resolution.
- To learn spatial dependencies through spatially-dense-temporally-sparse sequences that include both greyscale and corresponding RGB image frames to generate colorized versions of greyscale frames, thus, enhancing spatial resolution.

# Code Explanation:

### [data.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/data.py)
The `data.py` module is tasked with pre-processing and preparing the dataset consisting of 1-channel grayscale and 3-channel RGB TIF images. Here’s an in-depth look at the classes and functions:

- `CustomDataset` class:
  - __init__: Initializes the paths to grayscale and RGB directories (`grayscale_dir`, `rgb_dir`), the image size (`image_size`), and batch size (`batch_size`). It also includes a list of valid filename extensions (`valid_exts`) and transforms to apply to images before converting them to tensors.
  - __len__: Returns the number of images in the dataset.
  - __getitem__: Retrieves a grayscale and corresponding RGB image pair by index, applies the predefined transformations, and returns the tensor representations.
  - `get_autoencoder_batches`: Splits the dataset into training and validation sets for the AutoEncoder, and wraps them in DataLoader objects.
  - `get_lstm_batches`: Processes filenames to create sequential data for the LSTM model, respecting the sequence stride and length, and then splits into training and validation sets before creating their DataLoader objects.
  - `create_sequence_pairs`: Generates tuples of sequences with their corresponding target sequences, which are used as input and target for LSTM training.

### [autoencoder_model.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/autoencoder_model.py)
Defines `Grey2RGBAutoEncoder`, a deep learning PyTorch model comprising of a series of convolutional layers:

- `__init__`: Specifies the encoder and decoder components of the AutoEncoder architecture, containing stacks of `Conv2d` and `ConvTranspose2d` layers respectively, interleaved with `BatchNorm2d` and `LeakyReLU`, except the final layer (in Decoder only) which employs `Sigmoid`.
- `_make_layers`: Utility function to create sequential layers for either the encoder or decoder part of the AutoEncoder. This function builds layers based on the provided `channels` list, accommodating both `Conv2d` layers for the encoder and `ConvTranspose2d` layers for the decoder, ensuring the use of appropriate activation functions.
- `forward`: Defines the forward pass through both encoder and decoder components, outputting the reconstructed 3-channel RGB image from a single-channel grayscale image.

### [lstm_model.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/lstm_model.py)
Implements a Sequential model utilizing ConvLSTM cells for predicting interpolated frames:

- `ConvLSTMCell` class:
  - `__init__`: Sets up a convolutional layer (`self.conv`) with the appropriate number of input and output channels based on `input_dim`, `hidden_dim`, and `num_features`.
  - `forward`: Performs a forward pass through the ConvLSTM cell, generating the next hidden and cell states from the given input tensor and previous states.
- `ConvLSTM` class:
  - `__init__`: Constructs layers of ConvLSTM cells based on specified `input_dim`, `hidden_dims`, `kernel_size`, and `num_layers`. The `alpha` parameter is used to weight the input to each cell during sequence processing.
  - `init_hidden`: Initializes the hidden and cell states for all layers to zeros.
  - `forward`: Propagates a sequence of input tensors through the network, producing an output sequence and the final states. The sequence includes predictions for in-between frames, as well as an extra frame at the end.

### [losses.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/losses.py)
This module contains PyTorch loss functions:

- `LossMEP` class: Implements a composite loss which is a weighted combination of Mean Squared Error (MSE) and entropy:
  - `__init__`: Initializes the weighting factor `alpha`.
  - `forward`: Calculates MSE and entropy of the predictions, outputting the composite loss.
- `LossMSE` class: Represents a standard MSE loss for evaluating pixel-wise intensity differences.
- `SSIMLoss` class: Encapsulates the Structural Similarity Index Measure (SSIM) for assessing perceptual quality of images. The `forward` method averages the SSIM across all frames and constructs the loss as `1 - SSIM`.

### [main.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/main.py)
The `main.py` script orchestrates the entire process:

- Initializes the `CustomDataset` for handling input data.
- Prepares loss functions `LossMSE`, `LossMEP`, `SSIMLoss`.
- Declares AutoEncoder (`Grey2RGBAutoEncoder`) and LSTM (`ConvLSTM`) models.
- Forms four training configurations using combinations of loss functions and initializes trainer instances (`Trainer`) for each method.
- Executes the training loops for each configuration by calling `train_autoencoder` and `train_lstm` methods provided by the `Trainer` instances.
- Manages error handling, debugging prints, and model saving routines throughout the training process.

### [training.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/training.py)
The `Trainer` class encapsulates the setup and execution of training:

- `__init__`: Sets the device for training, prepares the model (potentially wrapped in `nn.DataParallel`), chooses the loss function and optimizer, and sets the model save path.
- `save_model`: Commits the model state to disk at the specified model save path.
- `train_autoencoder`: Executes training for the AutoEncoder model, with loops over epochs for both training and validation phases, managing forward and backward passes, optimizing parameters, and saving the best model based on the validation loss.
- `train_lstm`: Similar to `train_autoencoder`, it accommodates training for the LSTM model, handling sequences of images, and maintains the same logic for optimization and model saving based on validation loss.

## Contributions Welcome!
Your interest in contributing to the project is highly respected. Aiming for collaborative excellence, your insights, code improvements, and innovative ideas are highly appreciated. Make sure to check [Contributing Guidelines](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/CONTRIBUTING.md) for more information on how you can become an integral part of this project.

## Acknowledgements
A heartfelt thank you to all contributors and supporters who are on this journey to break new ground in video super-resolution technology.
