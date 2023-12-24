'''
Main Module
--------------------------------------------------------------------------------
'''

# Importing Custom Modules
from data import Dataset
from lstm_model import FrameInterpolationLSTM
from autoencoder_model import Grey2RGBAutoEncoder
from losses import LossMSE, LossMEP, SSIMLoss
from training import Trainer

# Import Necessary Libraries
import os

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
    print('Loading Dataset Initiated.')
    try:
        dataset = Dataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)
        print('Loading Dataset Completed.')
    except Exception as e:
        print(f"Loading Dataset In-Complete : \n{e}")

    # Import Loss Functions
    print('Importing Loss Functions Initiated.')
    loss_mse = LossMSE() # Mean Squared Error Loss
    loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss
    loss_ssim = SSIMLoss(data_range=1, size_average=True) # Structural Similarity Index Measure Loss
    print('Importing Loss Functions Complete.')

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split)
    print('AutoEncoder Model Data Initialized.')
    model_autoencoder = Grey2RGBAutoEncoder()
    print('AutoEncoder Model Initialized.')

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
    # Method 1 : Baseline : Mean Squared Error Loss for AutoEncoder and LSTM
    os.makedirs('../Models/Method1', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method1/model_autoencoder_m1.pth'
    trainer_autoencoder_baseline = Trainer(model_autoencoder, loss_mse, model_save_path_ae)
    print('Baseline AutoEncoder Trainer Initialized.')
    model_save_path_lstm = '../Models/model_lstm_m1.pth'
    trainer_lstm_baseline = Trainer(model_lstm, loss_mse, model_save_path_lstm)
    print('Baseline LSTM Trainer Initialized.')

    # Method 2 : Composite Loss for AutoEncoder and Mean Squared Error Loss for LSTM
    os.makedirs('../Models/Method2', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method2/model_autoencoder_m2.pth'
    trainer_autoencoder_m2 = Trainer(model_autoencoder, loss_mep, model_save_path_ae)
    print('Method-2 AutoEncoder Trainer Initialized.')
    print('Method-2 LSTM == Method-1 LSTM')

    # Method 3 : Mean Squared Error Loss for AutoEncoder and SSIM Loss for LSTM
    os.makedirs('../Models/Method3', exist_ok=True) # Creating Directory for Model Saving
    print('Method-3 AutoEncoder == Method-1 AutoEncoder')
    model_save_path_lstm = '../Models/Method3/model_lstm_m3.pth'
    trainer_lstm_m3 = Trainer(model_lstm, loss_ssim, model_save_path_lstm)
    print('Method-3 LSTM Trainer Initialized.')

    # Method 4 : Proposed Method : Composite Loss for AutoEncoder and SSIM Loss for LSTM
    print('Method-4 AutoEncoder == Method-2 AutoEncoder')
    print('Method-4 LSTM == Method-3 LSTM')


    '''
    Train Models, Obtain Trained Model
    ''' 
    # Method-1
    try:
        epochs = 5
        print('M1 AutoEncoder Training Start.')
        model_autoencoder_m1 = trainer_autoencoder_baseline.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        print('M1 AutoEncoder Training Complete.')
    except Exception as e:
        print(f"M1 AutoEncoder Training Error : \n{e}")
    try:
        epochs = 5
        print('M1 LSTM Training Start.')
        model_lstm_m1 = trainer_lstm_baseline.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)
        print('M1 LSTM Training Complete.') 
    except Exception as e:
        print(f"M1 LSTM Training Error : \n{e}")

    # Method-2
    try:
        epochs = 5
        print('M2 AutoEncoder Training Start.')
        model_autoencoder_m2 = trainer_autoencoder_m2.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        print('M2 AutoEncoder Training Complete.')
    except Exception as e:
        print(f"M2 AutoEncoder Training Error : \n{e}")
    # Method-2 LSTM == Method-1 LSTM, no need to train again

    # Method-3
    try:
        epochs = 5
        print('M3 LSTM Training Start.')
        model_lstm_m3 = trainer_lstm_m3.train_lstm(epochs, n_interpolate_frames, grey_sequence_train, grey_sequence_val)
        print('M3 LSTM Training Complete.') 
    except Exception as e:
        print(f"M3 LSTM Training Error : \n{e}")
    # Method-3 AutoEncoder == Method-1 AutoEncoder, no need to train again


if __name__ == '__main__':
    main()
