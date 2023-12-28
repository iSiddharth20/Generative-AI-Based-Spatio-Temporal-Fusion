'''
Module for Training Models
Separate Functions for Training AutoEncoder and LSTM Models
Saves the Model with the Lowest Validation Loss
--------------------------------------------------------------------------------
Forward Pass: Compute Output from Model from the given Input
Backward Pass: Compute the Gradient of the Loss with respect to Model Parameters
Initialize Best Validation Loss to Infinity as we will save model with lowest validation loss
'''

# Import Necessary Libraries
import torch

# Define Training Class
class Trainer():
    def __init__(self, model, loss_function, optimizer=None, model_save_path=None):
        print('Executing __init__ of Trainer Class from training.py')
        # Define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define the model and move it to the device
        self.model = model.to(self.device)
        # Define the loss function
        self.loss_function = loss_function
        # Define the optimizer
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Define the path to save the model
        self.model_save_path = model_save_path

    def save_model(self):
        print('Executing save_model of Trainer Class from training.py')
        # Save the model
        torch.save(self.model.state_dict(), self.model_save_path)

    def train_autoencoder(self, epochs, train_loader, val_loader):
        best_val_loss = float('inf')  
        for epoch in range(epochs):
            self.model.train()  # Set the Model to Training Mode
            # Training Loop
            for input, target in train_loader:  # Input - Grayscale Image, Target - RGB Image
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)  # Forward Pass
                loss = self.loss_function(output, target)  # Compute Training Loss
                self.optimizer.zero_grad()  # Zero gradients to prepare for Backward Pass
                loss.backward()  # Backward Pass
                self.optimizer.step()  # Update Model Parameters
            # Validation Loss Calculation
            self.model.eval()  # Set the Model to Evaluation Mode
            with torch.no_grad():  # Disable gradient computation
                val_loss = 0.0
                val_loss = sum(self.loss_function(self.model(input.to(self.device)), target.to(self.device)).item() for input, target in val_loader)  # Compute Total Validation Loss
                val_loss /= len(val_loader)  # Compute Average Validation Loss
            # Print epochs and losses
            print(f'AutoEncoder Epoch {epoch+1}/{epochs} --- Training Loss: {loss.item()} --- Validation Loss: {val_loss}')
            # If the current validation loss is lower than the best validation loss, save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update the best validation loss
                self.save_model()  # Save the model
        # Return the Trained Model
        return self.model

    def train_lstm(self, epochs, train_loader, val_loader):
        print('Executing train_lstm of Trainer Class from training.py')
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            # Training loop
            for input_sequence, target_sequence in train_loader:
                input_sequence, target_sequence = input_sequence.to(self.device), target_sequence.to(self.device)
                print(f'Input Sequence shape (before forward pass): {input_sequence.shape}')
                print(f'Target Sequence shape (before forward pass): {target_sequence.shape}')
                self.optimizer.zero_grad()  # Zero gradients
                output_sequence = self.model(input_sequence)  # Forward pass
                print(f'Output Sequence shape (after forward pass): {output_sequence.shape}')
                loss = self.loss_function(output_sequence, target_sequence)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update parameters
            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                val_loss = 0.0
                for input_sequence, target_sequence in val_loader:
                    input_sequence, target_sequence = input_sequence.to(self.device), target_sequence.to(self.device)
                    output_sequence = self.model(input_sequence)  # Forward pass
                    val_loss += self.loss_function(output_sequence, target_sequence).item()  # Accumulate loss
                val_loss /= len(val_loader)  # Average validation loss
            # Print epochs and losses
            print(f'Epoch {epoch+1}/{epochs} --- Training Loss: {loss.item()} --- Validation Loss: {val_loss}')
            # Model saving based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
        # Return the trained model
        return self.model