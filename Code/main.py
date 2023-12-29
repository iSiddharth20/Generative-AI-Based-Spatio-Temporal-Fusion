'''
Main Module
--------------------------------------------------------------------------------
'''

# Importing Custom Modules
from data import CustomDataset
from autoencoder_model import Grey2RGBAutoEncoder
from lstm_model import ConvLSTM
from losses import LossMSE, LossMEP, SSIMLoss
from training import Trainer


# Import Necessary Libraries
import os
import traceback
import torch

# Define Working Directories
grayscale_dir = '../Dataset/Greyscale'
rgb_dir = '../Dataset/RGB'

# Define Universal Parameters
image_height = 400
image_width = 600
batch_size = 2


def main():
    # Initialize Dataset Object (PyTorch Tensors)
    try:
        dataset = CustomDataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)
        print('Importing Dataset Complete.')
    except Exception as e:
        print(f"Importing Dataset In-Complete : \n{e}")
    # Import Loss Functions
    try:
        loss_mse = LossMSE() # Mean Squared Error Loss
        loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss
        loss_ssim = SSIMLoss() # Structural Similarity Index Measure Loss
        print('Importing Loss Functions Complete.')
    except Exception as e:
        print(f"Importing Loss Functions In-Complete : \n{e}")
    print('-'*20) # Makes Output Readable

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split=0.2)
    print('AutoEncoder Model Data Imported.')
    model_autoencoder = Grey2RGBAutoEncoder()
    print('AutoEncoder Model Initialized.')
    print('-'*20) # Makes Output Readable

    # Initialize LSTM Model and Import Dataloader (Training, Validation)
    data_lstm_train, data_lstm_val = dataset.get_lstm_batches(val_split=0.25, sequence_length=2)
    print('LSTM Model Data Imported.')
    model_lstm = ConvLSTM(input_dim=1, hidden_dims=[1,1,1], kernel_size=(3, 3), num_layers=3, alpha=0.5)
    print('LSTM Model Initialized.')
    print('-'*20) # Makes Output Readable

    '''
    Initialize Trainer Objects
    ''' 
    # Method 1 : Baseline : Mean Squared Error Loss for AutoEncoder and LSTM
    os.makedirs('../Models/Method1', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method1/model_autoencoder_m1.pth'
    trainer_autoencoder_baseline = Trainer(model_autoencoder, loss_mse, optimizer=torch.optim.Adam(model_autoencoder.parameters(), lr=0.001), model_save_path=model_save_path_ae)
    print('Method-1 AutoEncoder Trainer Initialized.')
    model_save_path_lstm = '../Models/Method1/model_lstm_m1.pth'
    trainer_lstm_baseline = Trainer(model_lstm, loss_mse, optimizer=torch.optim.Adam(model_lstm.parameters(), lr=0.001), model_save_path=model_save_path_lstm)
    print('Method-1 LSTM Trainer Initialized.')
    print('-'*10) # Makes Output Readable

    # Method 2 : Composite Loss (MSE + MaxEnt) for AutoEncoder and Mean Squared Error Loss for LSTM
    os.makedirs('../Models/Method2', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method2/model_autoencoder_m2.pth'
    trainer_autoencoder_m2 = Trainer(model=model_autoencoder, loss_function=loss_mep, optimizer=torch.optim.Adam(model_autoencoder.parameters(), lr=0.001), model_save_path=model_save_path_ae)
    print('Method-2 AutoEncoder Trainer Initialized.')
    print('Method-2 LSTM == Method-1 LSTM')
    print('-'*10) # Makes Output Readable

    # Method 3 : Mean Squared Error Loss for AutoEncoder and SSIM Loss for LSTM
    os.makedirs('../Models/Method3', exist_ok=True) # Creating Directory for Model Saving
    print('Method-3 AutoEncoder == Method-1 AutoEncoder')
    model_save_path_lstm = '../Models/Method3/model_lstm_m3.pth'
    trainer_lstm_m3 = Trainer(model_lstm, loss_ssim, optimizer=torch.optim.Adam(model_lstm.parameters(), lr=0.001), model_save_path=model_save_path_lstm)
    print('Method-3 LSTM Trainer Initialized.')
    print('-'*10) # Makes Output Readable

    # Method 4 : Proposed Method : Composite Loss (MSE + MaxEnt) for AutoEncoder and SSIM Loss for LSTM
    print('Method-4 AutoEncoder == Method-2 AutoEncoder')
    print('Method-4 LSTM == Method-3 LSTM')

    print('-'*20) # Makes Output Readable


    '''
    Train Models, Obtain Trained Model
    ''' 
    # Method-1
    try:
        epochs = 1
        print('Method-1 AutoEncoder Training Start')
        model_autoencoder_m1 = trainer_autoencoder_baseline.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        print('Method-1 AutoEncoder Training Complete.')
    except Exception as e:
        print(f"Method-1 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    print('-'*10) # Makes Output Readable
    try:
        epochs = 1
        print('Method-1 LSTM Training Start')
        model_lstm_m1 = trainer_lstm_baseline.train_lstm(epochs, data_lstm_train, data_lstm_val)
        print('Method-1 LSTM Training Complete.')
    except Exception as e:
        print(f"Method-1 LSTM Training Error : \n{e}")
        traceback.print_exc()
    print('-'*20) # Makes Output Readable

    # Method-2
    try:
        epochs = 1
        print('Method-2 AutoEncoder Training Start')
        model_autoencoder_m2 = trainer_autoencoder_m2.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        print('Method-2 AutoEncoder Training Complete.')
    except Exception as e:
        print(f"Method-2 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    print('-'*10) # Makes Output Readable
    print("Method-2 LSTM == Method-1 LSTM, No Need To Train Again.")
    print('-'*20) # Makes Output Readable

    # Method-3
    print("Method-3 AutoEncoder == Method-1 AutoEncoder, No Need To Train Again.")
    print('-'*10) # Makes Output Readable
    try:
        epochs = 1
        print('Method-3 LSTM Training Start.')
        model_lstm_m3 = trainer_lstm_m3.train_lstm(epochs, data_lstm_train, data_lstm_val)
        print('Method-3 LSTM Training Complete.')
    except Exception as e:
        print(f"Method-3 LSTM Training Error : \n{e}")
        traceback.print_exc()
    print('-'*20) # Makes Output Readable

    # Method-4
    print("Method-4 AutoEncoder == Method-2 AutoEncoder, No Need To Train Again.")
    print('-'*10) # Makes Output Readable
    print("Method-4 LSTM == Method-3 LSTM, No Need To Train Again.")
    print('-'*20) # Makes Output Readable


if __name__ == '__main__':
    main()
