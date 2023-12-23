'''
Module that specifies Model Evaluation and Analysis Techniques
--------------------------------------------------------------------------------
'''

import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

class Evaluator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def convert_to_image(self, tensor):
        tensor = tensor.clone()
        tensor = tensor * 255.0
        tensor = tensor.cpu().numpy().astype(np.uint8)
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor

    def evaluate(self, test_loader):
        self.model.eval()
        mse_total = 0
        ssim_total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_data = batch[0]
                output = self.model(batch_data)
                batch_data = self.convert_to_image(batch_data)
                output = self.convert_to_image(output)
                mse = ((batch_data - output) ** 2).mean(axis=None)
                mse_total += mse
                batch_ssim = np.mean([ssim(x, y, data_range=255) for x, y in zip(batch_data, output)])
                ssim_total += batch_ssim
        mse_avg = mse_total / len(test_loader)
        ssim_avg = ssim_total / len(test_loader)
        print('Test MSE: {:.4f}'.format(mse_avg))
        print('Test SSIM: {:.4f}'.format(ssim_avg))

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.show()