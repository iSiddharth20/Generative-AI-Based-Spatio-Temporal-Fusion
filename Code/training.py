'''
Module for Training Models
Seprate Functions for Training AutoEncoder and LSTM Models
Saves the Model with the Lowest Validation Loss
--------------------------------------------------------------------------------
Forward Pass: Compute Output from Model from the given Input
Backward Pass: Compute the Gradient of the Loss with respect to Model Parameters
Initialize Best Validation Loss to Inifnity as we will save model with lowest validation loss
'''

# Import Necessary Libraries
import torch

# Define Training Class
class Trainer():
    def __init__(self, model, loss_function, model_save_path):
        # Define the model
        self.model = model
        # Define the loss function
        self.loss_function = loss_function
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Define the path to save the model
        self.model_save_path = model_save_path

    def save_model(self):
        # Save the model
        torch.save(self.model.state_dict(), self.model_save_path)

    def train_autoencoder(self, epochs, train_loader, val_loader):
        self.model.train() # Set the Model to Training Mode
        best_val_loss = float('inf')  
        for epoch in range(epochs):
            # Training Loop
            for batch in train_loader:
                input, target = batch # Input - Grayscale Image, Target - RGB Image
                output = self.model(input)  # Forward Pass
                loss = self.loss_function(output, target)  # Compute Training Loss
                self.optimizer.zero_grad()  # Zero gradients to prepare for Backward Pass
                loss.backward()  # Backward Pass
                self.optimizer.step()  # Update Model Parameters
            # Validation Loss Calculation
            self.model.eval()  # Set the Model to Evaluation Mode
            with torch.no_grad():  # Disable gradient computation
                val_loss = 0  # Initialize the validation loss to 0
                # Loop over each batch from the validation set
                for batch in val_loader:
                    input, target = batch  # Unpack Batch
                    output = self.model(input)  # Forward Pass
                    loss = self.loss_function(output, target)  # Compute Loss
                    val_loss += loss.item()  # Compute Total Validation Loss
                val_loss /= len(val_loader)  # Compute Average Validation Loss
            # If the current validation loss is lower than the best validation loss, save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update the best validation loss
                self.save_model()  # Save the model
        # Return the Trained Model
        return self.model

    def train_lstm(self, epochs, n, image_sequence):
        # Write the code to train LSTM Model which generates intermediate frames between each pair of frames in the image sequence
        

