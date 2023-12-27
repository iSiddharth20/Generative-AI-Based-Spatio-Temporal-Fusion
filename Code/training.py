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

    def train_lstm(self, epochs, train_loader, val_loader):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            # Training loop
            for input_seqs, target_seqs in train_loader:
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)

                # Perform forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_seqs)

                # Compute loss; assuming that we are using an unsupervised learning approach for now
                loss = self.loss_function(outputs, target_seqs)
                train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # Average the loss over all batches and print
            train_loss /= len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')

            # Validation loop
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for input_seqs, target_seqs in val_loader:
                    input_seqs, target_seqs = input_seqs.to(self.device), target_seqs.to(self.device)

                    # Perform validation forward pass
                    outputs = self.model(input_seqs)

                    # Compute loss; assuming that we are using an unsupervised learning approach for now
                    loss = self.loss_function(outputs, target_seqs)
                    val_loss += loss.item()

            val_loss /= len(val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}')

            # Save model with best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                print('Best model saved with validation loss: {:.4f}'.format(val_loss))

        return self.model