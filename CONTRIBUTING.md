# Contributing to [Generative-AI-Based-Spatio-Temporal-Fusion](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion)

Thank you for considering contributing to [Generative-AI-Based-Spatio-Temporal-Fusion](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion)! Your efforts and contributions are crucial for the success of this project and greatly appreciated. Below, you'll find the process and guidelines for contributing.

## This File will be Updated with the New Code Structure Details by December 14, 2023. Thank You!

## Guidelines:

### Pull Requests
- 🍴 Fork the repository.
- 📌 Include descriptive commit messages.

### Code Styleguide
- 💬 Include comments explaining why certain pieces of code were implemented.
- ✅ Write tests (if applicable) for the new code you're submitting.

## 🙌 Acknowledgments
Thanks to all the contributors who have helped this project grow!

# Required Codebase:

### [LSTM.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/LSTM.py)
- Define a PyTorch LSTM model class for frame interpolation, generating an entire greyscale image for a given sequence. The model takes a sequence (sequence length = `len_seq`) of grayscale images (400x600) as input and predicts the following, according to user preference:
	- The next image in the sequence.
	- `n` images interpolated between existing images of the sequence.
- Write a function using PyTorch to perform hyperparameter tuning for an LSTM model, testing various learning rates and numbers of hidden units. Record the results of hyperparameter tuning, i.e., the performance of each parameter combination.

### [AutoEncoder.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/AutoEncoder.py)
- Define a PyTorch AutoEncoder model class with:
	- An encoder that maps 400x600 greyscale images to known RGB images.
	- A decoder that reconstructs the RGB images from the greyscale images.

### [LossFunction.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/LossFunction.py)
- Write a PyTorch loss function named `loss_MEP` that combines Mean Squared Error with a Maximum Entropy regularization term for an AutoEncoder.
	- The Composite loss function (loss_MEP) is given by:
		` L = (1/2) * Σ(i=1 to N) (x_i - x̂_i)^2 - λmep * H(q(z|x) `
		where:
		- L represents the Composite Loss Function.
		- N is the number of dimensions in the latent space.
		- x_i is the input data of the AutoEncoder (greyscale image).
		- x̂_i is the output data of the AutoEncoder (RGB image).
		- λmep is a Maximum Entropy regularization parameter.
		- H(q(z|x)) represents the entropy of the variational posterior distribution q(z|x).
- Write a PyTorch loss function named `loss_MLP` that combines Mean Squared Error with a Maximum Likelihood regularization term for an AutoEncoder.
	- The Composite loss function (loss_MLP) is given by:
		` L = (1/2) * Σ(i=1 to N) (x_i - x̂_i)^2 + λmlp) `
		where:
		- L represents the Composite Loss Function.
		- N is the number of dimensions in the latent space.
		- x_i is the input data of the AutoEncoder (greyscale image).
		- x̂_i is the output data of the AutoEncoder (RGB image).
		- λmlp is a Maximum Likelihood regularization parameter.

### [main.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/main.py)
- Write a Python function using PyTorch to load a dataset of grayscale TIF images from directory `../Dataset/Grey` and RGB TIF images from directory `../Dataset/RGB`, resize them to 400x600 pixels, and normalize the pixel values (0-255).
- Split the dataset into training, testing, and validation sets using `sklearn.train_test_split` with a ratio of 60:20:20 and convert to PyTorch tensors using batch size = `batch_size`.
- Export the Training, Testing, and Validation Sets to the directory `../Dataset/PyTorchTensors` using `torch.save`.
- Import the LSTM model class from lstm.py.
- Import the AutoEncoder model class from autoencoder.py.
- Outline a training loop (EPOCHS = `num_epochs`) in PyTorch that trains an LSTM and an AutoEncoder model using the Adam optimizer, and include calculating and printing the loss every epoch.
- Train the model named `model_MEP` using `loss_MEP` as the Loss Function.
- Train the model named `model_MLP` using `loss_MLP` as the Loss Function.
- Export the Trained Models to the directory `../TrainedModel` if the new model has lower loss than previous one in thae training loop.

### [Results.py](https://github.com/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion/blob/main/Code/Results.py)
- Import the Validation Sets from the directory `../Dataset/PyTorchTensors`.
- Import the Trained Models.
- Implement a PyTorch validation loop that computes the Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM) as validation metrics on Validation Sets of greyscale and RGB image pairs.
- Create a Python function using PyTorch to compare the performance of two models (`model_MEP` and `model_MLP`) trained with different regularization principles: Maximum Likelihood and Maximum Entropy.
