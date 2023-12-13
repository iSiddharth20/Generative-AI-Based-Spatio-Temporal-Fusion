'''
Module that specifies Training Methodology
--------------------------------------------------------------------------------
'''

import torch
from torch import optim
from sklearn.model_selection import KFold
from copy import deepcopy

class Trainer:
    def __init__(self, model, loss_fn, optimizer=None, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)  # LR scheduler
        self.kf = KFold(n_splits=10)
        self.best_model = None
        self.best_loss = float('inf')

    def train(self, epochs, train_loader, val_loader=None):
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()
                total_train_loss += loss.item()
            
            self.scheduler.step()  # Step the LR scheduler
                       
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            # Validation phase
            if val_loader:
                total_val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

                # Save the best model based on validation loss
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self.best_model = deepcopy(self.model)
                    print("Validation loss decreased, saving new best model...")

            # Code for Early Stopping
            if avg_val_loss > prev_val_loss:
                break
            prev_val_loss = avg_val_loss


    def get_best_model(self):
        return self.best_model if self.best_model else self.model