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
image_height = 4000
image_width = 6000
batch_size = 4
val_split = 0.2


def main():
    # Initialize Dataset Object (PyTorch Tensors)
    dataset = Dataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)

    # Import Loss Functions
    loss_mlp = LossMLP(alpha=0.4) # Maximum Likelihood Loss
    loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    model_autoencoder = Grey2RGBAutoEncoder()
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split)

    # Initialize LSTM Model and Import Image Sequences (Training, Validation)
    grey_sequence_train, grey_sequence_val = dataset.get_lstm_batches()
    C = 1 
    hidden_size = 64
    num_layers = 3
    n_interpolate_frames = 1 # Number of intermediate frames to interpolate
    kernel_size = (3, 3)
    model_lstm = FrameInterpolationLSTM(C, hidden_size, kernel_size, num_layers)

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
    model_lstm_mlp = trainer_mlp_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)

    # Maximize Entropy Principle
    epochs = 5
    model_autoencoder_mep = trainer_mep_autoencoder.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
    model_lstm_mep = trainer_mep_lstm.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)

if __name__ == '__main__':
    main()
