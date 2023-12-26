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
    def __init__(self, model, loss_function, model_save_path):
        # Define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define the model and move it to the device
        self.model = model.to(self.device)
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
        best_val_loss = float('inf')  
        for epoch in range(epochs):
            self.model.train()  # Set the Model to Training Mode
            # Training Loop
            for input, target in train_loader:  # Input - Grayscale Image, Target - RGB Image
                input, target = input.to(self.device), target.to(self.device)
                print(f'Input shape: {input.shape}, Target shape: {target.shape}')
                output = self.model(input)  # Forward Pass
                loss = self.loss_function(output, target)  # Compute Training Loss
                print(f'Epoch: {epoch}, Training loss: {loss.item()}')
                self.optimizer.zero_grad()  # Zero gradients to prepare for Backward Pass
                loss.backward()  # Backward Pass
                self.optimizer.step()  # Update Model Parameters
            # Validation Loss Calculation
            self.model.eval()  # Set the Model to Evaluation Mode
            with torch.no_grad():  # Disable gradient computation
                val_loss = 0.0
                val_loss = sum(self.loss_function(self.model(input.to(self.device)), target.to(self.device)).item() for input, target in val_loader)  # Compute Total Validation Loss
                val_loss /= len(val_loader)  # Compute Average Validation Loss
            # Print the epoch number and the validation loss
            print(f'Epoch : {epoch}, Validation Loss : {val_loss}')
            # If the current validation loss is lower than the best validation loss, save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update the best validation loss
                self.save_model()  # Save the model
        # Return the Trained Model
        return self.model

    def train_lstm(self, epochs, n_interpolate_frames, train_data, val_data):
        min_val_loss = float('inf')  # Initialize the minimum validation loss to infinity
        
        for epoch in range(epochs):
            print(f'Epoch: {epoch}, Training')
            self.model.train()  # Set the model to training mode
            train_loss = 0.0
            
            # Training Loop
            for sequences, targets in train_data:
                print(f'Input sequence shape: {sequences.shape}, Target sequence shape: {targets.shape}')
                self.optimizer.zero_grad()  # Reset the gradients accumulated from the previous iteration
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(sequences, n_interpolate_frames)
                # sequences.requires_grad_()  # Ensure gradients are required for the input to ConvLSTM
                # targets.requires_grad_(False)  # Ensure targets do not require gradients
                # Assuming the outputs and targets are of shape [batch_size, seq_len, channels, height, width]
                # Compute Training Loss only on the interpolated frames (not on the original frames)
                loss = self.loss_function(outputs[:, 1:-1], targets[:, 1:-1])
                print(f'Train loss: {loss.item()}')
                loss.backward()  # Backward Pass
                self.optimizer.step()  # Update Model Parameters
                train_loss += loss.item()
            
            train_loss /= len(train_data)
            print(f'Epoch : {epoch}, Training Loss : {train_loss}')

            # Validation Loss Calculation
            self.model.eval()  # Set the Model to Evaluation Mode
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in val_data:
                    print(f'Epoch: {epoch}, Validation')
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(sequences, n_interpolate_frames)
                    # Compute Validation Loss only on interpolated frames (not on the original frames)
                    predicted_interpolated_frames = outputs[:, 1:-1].reshape(-1, *outputs.shape[2:])  # Reshape to (B * seq_len, C, H, W)
                    true_interpolated_frames = targets[:, 1:-1].reshape(-1, *targets.shape[2:])  # Same reshaping for targets
                    loss = self.loss_function(predicted_interpolated_frames, true_interpolated_frames)
                    val_loss += loss.item()
                
                val_loss /= len(val_data)
                print(f'Validation loss: {val_loss}')
            
            # Print the epoch number and the validation loss
            print(f'Epoch : {epoch}, Validation Loss : {val_loss}')

            # If the current validation loss is lower than the best validation loss, save the model
            if val_loss < min_val_loss:
                min_val_loss = val_loss  # Update the best validation loss
                self.save_model()  # Save the model
                
        # Return the Trained Model
        return self.model