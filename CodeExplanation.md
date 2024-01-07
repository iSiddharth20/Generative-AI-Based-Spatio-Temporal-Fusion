# GenAI-Powered Spatio-Temporal Fusion for Video Super-Resolution

#### Based on PyTorch, Install [Here](https://pytorch.org/get-started/locally/)
#### Access Latest Updates [Here](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion)


## Introduction
This project is a novel approach to enhance video resolution both spatially and temporally using generative AI techniques. By leveraging Auto-Encoders and LSTM Networks, the project aims to interpolate high-temporal-resolution grayscale images and colorize them by learning from a corresponding set of RGB images, ultimately achieving high-fidelity video super-resolution.


## Research Objective
The main goals of the project are:
- To learn temporal dependencies among spatially-sparse-temporally-dense greyscale image frames to predict and interpolate new frames, hence, increasing temporal resolution.
- To learn spatial dependencies through spatially-dense-temporally-sparse sequences that include both greyscale and corresponding RGB image frames to generate colorized versions of greyscale frames, thus, enhancing spatial resolution.

## What is New in this Version?
- Flexibility to Add different independent Datasets for AutoEncoder and LSTM, now these components don't necessarily have to be trained on the same images.
- Ability to define independent and different batch size for AutoEncoder Training and LSTM Training.
- SSIM will now act as Regularization Term along with MSE in SSIMLoss Function.
- Updated and Corrected Math for MaxEnt Regularization Term.
- For LSTM Training, Each Individual Sequence will now be split to create Training/Validation Datasets.


## Code Structure Overview
The codebase is organized into several Python modules, each serving a distinct purpose in the project pipeline. Here's a broader overview of the file structure and functionality:

```
├── Code/
│   ├── data.py              # Dataset preparation and data loader definitions
│   ├── main.py              # Orchestrator for initializing and training models
│   ├── training.py          # Defines the Trainer class for model training
│   ├── autoencoder_model.py # Contains the AutoEncoder architecture
│   ├── lstm_model.py        # Defines the LSTM architecture for frame interpolation
│   └── losses.py            # Custom loss functions utilized in training
```

- **data.py**: This script is the starting point for data pipeline operations. The `CustomDataset` class inherits from `torch.utils.data.Dataset` and implements methods for data preparation, including `__getitem__` for lazy loading. It utilizes PIL for image manipulations and `torchvision.transforms` to resize images and convert them to PyTorch tensors. It’s essential for coders to ensure the correct data directory paths and acceptable image extensions are specified to avoid loading issues.

- **main.py**: The epicenter of model execution, this script employs PyTorch's distributed computing features when training on multiple GPUs. The `main_worker` function distributes GPU workload among parallel processes. It invokes `main` function, which instantiates the models, sets up training data, initializes loss functions, and loops through various training configurations, one for each set of loss function combinations.

- **training.py**: Within this file, the `Trainer` class manages the training loops. Coders should pay attention to the `train_autoencoder` and `train_lstm` functions, each tailored specifically for its respective model. These functions utilize PyTorch's automatic differentiation mechanism for gradient computation (`backward()`) and apply optimizer steps (`step()`) to update the model weights. The code supports distributed training using `DistributedDataParallel`, and it is crucial to manage device assignments correctly to avoid device misalignment issues.

- **autoencoder_model.py**: Coders can find the model definition of the AutoEncoder in `Grey2RGBAutoEncoder`, which uses a typical encoder-decoder structure with a series of `nn.Conv2d` and `nn.ConvTranspose2d` paired with batch normalization and activation functions. The final sigmoid activation function in the decoder guarantees the output image's pixel values range between 0 and 1.

- **lstm_model.py**: Here, `ConvLSTM` and `ConvLSTMCell` classes implement the components of a convolutional LSTM network capable of handling spatial-temporal data. The `ConvLSTMCell` performs gated operations using convolutional layers, while `ConvLSTM` manages temporal sequences and predicts intermediate frames. Coders intending to enhance this functionality should have a firm grasp on sequence processing and recurrent neural network principles.

- **losses.py**: Loss functions are defined as classes, inheriting from `nn.Module`. The `LossMSE` and `SSIMLoss` are standard while `LossMEP` introduces a custom composite loss involving a maximum entropy regularization term. The novelty here lies in the balancing act performed using an `alpha` parameter, which controls the trade-off between fidelity (MSE) and diversity (entropy). This is a key area for coders looking to experiment with loss function formulation and its effects on training dynamics.


### Navigating the Code for Development
To effectively navigate and contribute to the codebase, it's recommended that coders:
1. Begin by examining `main.py` to understand the orchestration logic and to gather insights into how the different modules fit into the broader workflow.
2. Delve into `data.py` to understand the dataset structure expected by model training routines and how data augmentation is achieved through transformations.
3. Explore the model definitions (`autoencoder_model.py` and `lstm_model.py`) to comprehend the network architectures or to modify them for experimental purposes.
4. Study `training.py` to grasp the training loops and mechanism utilized for the two types of models. Any enhancements in training procedures, optimization, or logging should happen here.
5. Assess and potentially refine the loss functions (`losses.py`) for improved model performance or to execute novel training strategies.


## Contributions Welcome!
Your interest in contributing to the project is highly respected. Aiming for collaborative excellence, your insights, code improvements, and innovative ideas are highly appreciated. Make sure to check [Contributing Guidelines](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/CONTRIBUTING.md) for more information on how you can become an integral part of this project.


## Acknowledgements
A heartfelt thank you to all contributors and supporters who are on this journey to break new ground in video super-resolution technology.

