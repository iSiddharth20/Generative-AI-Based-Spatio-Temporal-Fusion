'''
Module that Combines AutoEncoder, LSTM to Train the Model using 2 Kinds of loss Functions.
--------------------------------------------------------------------------------
This is a Template Code and Needs to be Modified based on the Problem Statement 
'''

# Importing Necessary Libraries and Individual Components

import torch
import torch.optim as optim

from LossFunction import CompositeLossFunction, RegularLossFunction
from AutoEncoder import AutoEncoder
from LSTM import LSTMModule
import pytorch_ssim


# Initialize the model components
autoencoder = AutoEncoder(input_size=000, hidden_size=000, output_size=000)
lstm = LSTMModule(input_size=000, hidden_size=000, output_size=000, num_layers=000, dropout=000)

# Initialize the Composite Loss Function (Alpha may need tuning)
loss_MEP = CompositeLossFunction(alpha=0.5) 

# Initialize the Regular Loss Function (Alpha may need tuning)
loss_MLP = RegularLossFunction(alpha=0.5) 

# Initialize Adam Optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Initialize the Training Loop
def train_model(autoencoder, lstm, loss_function, optimizer, train_loader, device, model_name):
    lstm.train()
    autoencoder.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send data and target to device
        data, target = data.to(device), target.to(device)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        # Pass the data through the LSTM
        output = lstm(data)
        # Pass the output through the AutoEncoder
        output = autoencoder(output)
        # Calculate the loss
        loss = loss_function(output, target)
        # Backpropagate the loss
        loss.backward()
        # Update the weights
        optimizer.step()
        # Update the training loss
        train_loss += loss.item()
    # Return the training loss
    return train_loss


# Implement a PyTorch validation loop that computes the Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM)
def validate_model(autoencoder, lstm, val_loader, device, loss_function):
    lstm.eval()
    autoencoder.eval()
    val_loss = 0
    mse = 0
    ssim = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # Send data and target to device
            data, target = data.to(device), target.to(device)
            # Pass the data through the LSTM
            output = lstm(data)
            # Pass the output through the AutoEncoder
            output = autoencoder(output)
            # Calculate the loss
            loss = loss_function(output, target)
            # Update the validation loss
            val_loss += loss.item()
            # Calculate the MSE
            mse += torch.mean((output - target) ** 2)
            # Calculate the SSIM
            ssim += pytorch_ssim.ssim(output, target)
    # Return the validation loss, MSE and SSIM
    return val_loss, mse, ssim
