'''
Main Module
--------------------------------------------------------------------------------
'''

# Importing Custom Modules
from data import Dataset
from lstm_model import FrameInterpolationLSTM
from autoencoder_model import Grey2RGBAutoEncoder
from losses import LossMLP, LossMEP
from training import Trainer

# Import Necessary Libraries
import os
import torch
import numpy as np
from PIL import Image

# Define Working Directories
grayscale_dir = '../Dataset/Greyscale'
rgb_dir = '../Dataset/RGB'

# Define Universal Parameters
image_height = 4000
image_width = 6000
batch_size = 4
val_split = 0.2


def generate_rgb_sequence(model_lstm, model_autoencoder, grey_sequence, n_interpolate_frames, 
                          model_save_path_lstm, model_save_path_ae, generated_sequence_dir):

    if os.path.exists(model_save_path_lstm):
        model_lstm.load_state_dict(torch.load(model_save_path_lstm))
        model_lstm.eval()

    if os.path.exists(model_save_path_ae):
        model_autoencoder.load_state_dict(torch.load(model_save_path_ae))
        model_autoencoder.eval()


    full_sequence_gray = model_lstm(grey_sequence, n_interpolate_frames)


    full_sequence_rgb = []
    with torch.no_grad():
        for i in range(full_sequence_gray.size(1)): 
            gray_frame = full_sequence_gray[:, i, :, :]
            rgb_frame = model_autoencoder(gray_frame.unsqueeze(dim=0))
            full_sequence_rgb.append(rgb_frame)


    os.makedirs(generated_sequence_dir, exist_ok=True)
    for idx, rgb_tensor in enumerate(full_sequence_rgb):

        image_data = rgb_tensor.squeeze().cpu().numpy()
        image_data = np.transpose(image_data, (1, 2, 0)) 
        image_data = (image_data * 255).astype(np.uint8)
        image = Image.fromarray(image_data)

        image_path = os.path.join(generated_sequence_dir, f'generated_frame_{idx:04d}.tif')
        image.save(image_path)

    print('The generated sequence of RGB images has been saved.')


def main():
    # Initialize Dataset Object (PyTorch Tensors)
    dataset = Dataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)

    # Import Loss Functions
    loss_mlp = LossMLP(alpha=0.4) # Maximum Likelihood Loss
    loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    model_autoencoder = Grey2RGBAutoEncoder()
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split) # ??

    # Initialize LSTM Model and Import Image Sequences (Training, Validation)
    grey_sequence = dataset.get_lstm_batches()
    C, H, W = 1, image_height, image_width 
    hidden_size = 64
    num_layers = 3
    n_interpolate_frames = 1 # Number of intermediate frames to interpolate
    kernel_size = (3, 3)
    model_lstm = FrameInterpolationLSTM(C, hidden_size, kernel_size, num_layers, C)

    '''
    Initialize Trainer Objects
    ''' 
    # Maximize Likelihood Principle
    model_save_path_ae = '../Models/model_autoencoder_mlp.pth'
    trainer_mlp_autoencoder = Trainer(model_autoencoder, loss_mlp, model_save_path_ae)
    model_save_path_lstm = '../Models/model_lstm_mlp.pth'
    trainer_mlp_lstm = Trainer(model_lstm, loss_mlp, model_save_path_lstm)
    # Maximize Entropy Principle
    model_save_path_ae = '../Models/model_autoencoder_mep.pth'
    trainer_mep_autoencoder = Trainer(model_autoencoder, loss_mep, model_save_path_ae)
    model_save_path_lstm = '../Models/model_lstm_mep.pth'
    trainer_mep_lstm = Trainer(model_lstm, loss_mep, model_save_path_lstm)

    '''
    Train Models, Obtain Trained Model
    ''' 
    # Maximize Likelihood Principle
    epochs = 5
    model_autoencoder_mlp = trainer_mlp_autoencoder.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val) 
    model_lstm_mlp = trainer_mlp_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence)

    # Maximize Entropy Principle
    epochs = 5
    model_autoencoder_mep = trainer_mep_autoencoder.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
    model_lstm_mep = trainer_mep_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence)

    '''
    Pass Output of LSTM Model to AutoEncoder Model to Obtain Final Output
    '''
    # Maximize Likelihood Principle
    model_save_path_ae = '../Models/model_autoencoder_mlp.pth'
    model_save_path_lstm = '../Models/model_lstm_mlp.pth'
    generated_sequence_dir = '../Dataset/GeneratedSequence/MLP'
    generate_rgb_sequence(model_lstm_mlp, model_autoencoder_mlp, grey_sequence, n_interpolate_frames, 
                          model_save_path_lstm, model_save_path_ae, generated_sequence_dir)

    # Maximize Entropy Principle
    model_save_path_ae = '../Models/model_autoencoder_mep.pth'
    model_save_path_lstm = '../Models/model_lstm_mep.pth'
    generated_sequence_dir = '../Dataset/GeneratedSequence/MEP'
    generate_rgb_sequence(model_lstm_mep, model_autoencoder_mep, grey_sequence, n_interpolate_frames, 
                          model_save_path_lstm, model_save_path_ae, generated_sequence_dir)


if __name__ == '__main__':
    main()




