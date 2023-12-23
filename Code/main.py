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

# Define Working Directories
grayscale_dir = '../Dataset/Greyscale'
rgb_dir = '../Dataset/RGB'

# Define Universal Parameters
image_height = 400
image_width = 600
batch_size = 4
val_split = 0.2


def main():
    # Initialize Dataset Object (PyTorch Tensors)
    dataset = Dataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)
    print('Loading Dataset Completed.')

    # Import Loss Functions
    loss_mlp = LossMLP(alpha=0.4) # Maximum Likelihood Loss
    loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    model_autoencoder = Grey2RGBAutoEncoder()
    print('AutoEncoder Model Initialized.')
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split)
    print('AutoEncoder Model Data Initialized.')

    # Initialize LSTM Model and Import Image Sequences (Training, Validation)
    grey_sequence_train, grey_sequence_val = dataset.get_lstm_batches()
    print('LSTM Model Data Initialized.')
    C = 1 
    hidden_size = 64
    num_layers = 3
    n_interpolate_frames = 1 # Number of intermediate frames to interpolate
    kernel_size = (3, 3)
    model_lstm = FrameInterpolationLSTM(C, hidden_size, kernel_size, num_layers)
    print('LSTM Model Initialized.')

    '''
    Initialize Trainer Objects
    ''' 
    # Maximize Likelihood Principle
    model_save_path_ae = '../Models/model_autoencoder_mlp.pth'
    trainer_mlp_autoencoder = Trainer(model_autoencoder, loss_mlp, model_save_path_ae)
    print('AutoEncoder MLP Trainer Initialized.')
    model_save_path_lstm = '../Models/model_lstm_mlp.pth'
    trainer_mlp_lstm = Trainer(model_lstm, loss_mlp, model_save_path_lstm)
    print('LSTM MLP Trainer Initialized.')
    # Maximize Entropy Principle
    model_save_path_ae = '../Models/model_autoencoder_mep.pth'
    trainer_mep_autoencoder = Trainer(model_autoencoder, loss_mep, model_save_path_ae)
    print('AutoEncoder MEP Trainer Initialized.')
    model_save_path_lstm = '../Models/model_lstm_mep.pth'
    trainer_mep_lstm = Trainer(model_lstm, loss_mep, model_save_path_lstm)
    print('LSTM MEP Trainer Initialized.')

    '''
    Train Models, Obtain Trained Model
    ''' 
    # Maximize Likelihood Principle
    epochs = 5
    print('AutoEncoder MLP Training Initialized.')
    model_autoencoder_mlp = trainer_mlp_autoencoder.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
    print('AutoEncoder MLP Training Complete.')
    print('LSTM MLP Training Initialized.') 
    model_lstm_mlp = trainer_mlp_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)
    print('LSTM MLP Training Complete.') 

    # Maximize Entropy Principle
    epochs = 5
    print('AutoEncoder MEP Training Initialized.') 
    model_autoencoder_mep = trainer_mep_autoencoder.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
    print('AutoEncoder MEP Training Complete.') 
    print('LSTM MEP Training Initialized.') 
    model_lstm_mep = trainer_mep_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)
    print('LSTM MEP Training Complete.') 

if __name__ == '__main__':
    main()
