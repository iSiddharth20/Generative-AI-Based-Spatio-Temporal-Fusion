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
import torch.multiprocessing as mp
import torch.distributed as dist

# Define Working Directories
grayscale_dir = '../Dataset/Greyscale'
rgb_dir = '../Dataset/RGB'

# Define Universal Parameters
image_height = 400
image_width = 600
batch_size = 2

def main_worker(rank, world_size):
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    # Initialize the distributed environment.
    torch.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    main(rank)  # Call the existing main function.

def main(rank):
    # Initialize Dataset Object (PyTorch Tensors)
    try:
        dataset = CustomDataset(grayscale_dir, rgb_dir, (image_height, image_width), batch_size)
        if rank == 0:
            print('Importing Dataset Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Importing Dataset In-Complete : \n{e}")
    if rank == 0:
        print('-'*20) # Makes Output Readable
    # Import Loss Functions
    try:
        loss_mse = LossMSE() # Mean Squared Error Loss
        loss_mep = LossMEP(alpha=0.4) # Maximum Entropy Loss
        loss_ssim = SSIMLoss() # Structural Similarity Index Measure Loss
        if rank == 0:
            print('Importing Loss Functions Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Importing Loss Functions In-Complete : \n{e}")
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    data_autoencoder_train, data_autoencoder_val = dataset.get_autoencoder_batches(val_split=0.2)
    if rank == 0:
        print('AutoEncoder Model Data Imported.')
    model_autoencoder = Grey2RGBAutoEncoder()
    if rank == 0:
        print('AutoEncoder Model Initialized.')
        print('-'*20) # Makes Output Readable

    # Initialize LSTM Model and Import Dataloader (Training, Validation)
    data_lstm_train, data_lstm_val = dataset.get_lstm_batches(val_split=0.25, sequence_length=2)
    if rank == 0:
        print('LSTM Model Data Imported.')
    model_lstm = ConvLSTM(input_dim=1, hidden_dims=[1,1,1], kernel_size=(3, 3), num_layers=3, alpha=0.5)
    if rank == 0:
        print('LSTM Model Initialized.')
        print('-'*20) # Makes Output Readable

    '''
    Initialize Trainer Objects
    ''' 
    # Method 1 : Baseline : Mean Squared Error Loss for AutoEncoder and LSTM
    os.makedirs('../Models/Method1', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method1/model_autoencoder_m1.pth'
    trainer_autoencoder_baseline = Trainer(model=model_autoencoder, 
                                           loss_function=loss_mse, 
                                           optimizer=torch.optim.Adam(model_autoencoder.parameters(), lr=0.001), 
                                           model_save_path=model_save_path_ae, 
                                           rank=rank)
    if rank == 0:
        print('Method-1 AutoEncoder Trainer Initialized.')
    model_save_path_lstm = '../Models/Method1/model_lstm_m1.pth'
    trainer_lstm_baseline = Trainer(model=model_lstm, 
                                    loss_function=loss_mse, 
                                    optimizer=torch.optim.Adam(model_lstm.parameters(), lr=0.001), 
                                    model_save_path=model_save_path_lstm, 
                                    rank=rank)
    if rank == 0:
        print('Method-1 LSTM Trainer Initialized.')
        print('-'*10) # Makes Output Readable

    # Method 2 : Composite Loss (MSE + MaxEnt) for AutoEncoder and Mean Squared Error Loss for LSTM
    os.makedirs('../Models/Method2', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method2/model_autoencoder_m2.pth'
    trainer_autoencoder_m2 = Trainer(model=model_autoencoder, 
                                     loss_function=loss_mep, 
                                     optimizer=torch.optim.Adam(model_autoencoder.parameters(), lr=0.001), 
                                     model_save_path=model_save_path_ae, 
                                     rank=rank)
    if rank == 0:
        print('Method-2 AutoEncoder Trainer Initialized.')
        print('Method-2 LSTM == Method-1 LSTM')
        print('-'*10) # Makes Output Readable

    # Method 3 : Mean Squared Error Loss for AutoEncoder and SSIM Loss for LSTM
    os.makedirs('../Models/Method3', exist_ok=True) # Creating Directory for Model Saving
    if rank == 0:
        print('Method-3 AutoEncoder == Method-1 AutoEncoder')
    model_save_path_lstm = '../Models/Method3/model_lstm_m3.pth'
    trainer_lstm_m3 = Trainer(model=model_lstm, 
                              loss_function=loss_ssim, 
                              optimizer=torch.optim.Adam(model_lstm.parameters(), lr=0.001), 
                              model_save_path=model_save_path_lstm, 
                              rank=rank)
    if rank == 0:
        print('Method-3 LSTM Trainer Initialized.')
        print('-'*10) # Makes Output Readable

    # Method 4 : Proposed Method : Composite Loss (MSE + MaxEnt) for AutoEncoder and SSIM Loss for LSTM
    if rank == 0:
        print('Method-4 AutoEncoder == Method-2 AutoEncoder')
        print('Method-4 LSTM == Method-3 LSTM')
        print('-'*20) # Makes Output Readable


    '''
    Train Models, Obtain Trained Model
    ''' 
    # Method-1
    try:
        epochs = 1
        if rank == 0:
            print('Method-1 AutoEncoder Training Start')
        model_autoencoder_m1 = trainer_autoencoder_baseline.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        if rank == 0:
            print('Method-1 AutoEncoder Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-1 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            trainer_autoencoder_baseline.cleanup_ddp()
    if rank == 0:
        print('-'*10) # Makes Output Readable
    try:
        epochs = 1
        if rank == 0:
            print('Method-1 LSTM Training Start')
        model_lstm_m1 = trainer_lstm_baseline.train_lstm(epochs, data_lstm_train, data_lstm_val)
        if rank == 0:
            print('Method-1 LSTM Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-1 LSTM Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            trainer_lstm_baseline.cleanup_ddp()
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Method-2
    try:
        epochs = 1
        if rank == 0:
            print('Method-2 AutoEncoder Training Start')
        model_autoencoder_m2 = trainer_autoencoder_m2.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        if rank == 0:
            print('Method-2 AutoEncoder Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-2 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    finally:
        trainer_autoencoder_m2.cleanup_ddp()
    if rank == 0:
        print('-'*10) # Makes Output Readable
        print("Method-2 LSTM == Method-1 LSTM, No Need To Train Again.")
        print('-'*20) # Makes Output Readable

    # Method-3
    if rank == 0:
        print("Method-3 AutoEncoder == Method-1 AutoEncoder, No Need To Train Again.")
        print('-'*10) # Makes Output Readable
    try:
        epochs = 1
        if rank == 0:
            print('Method-3 LSTM Training Start.')
        model_lstm_m3 = trainer_lstm_m3.train_lstm(epochs, data_lstm_train, data_lstm_val)
        if rank == 0:
            print('Method-3 LSTM Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-3 LSTM Training Error : \n{e}")
        traceback.print_exc()
    finally:
        trainer_lstm_m3.cleanup_ddp()
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Method-4
    if rank == 0:
        print("Method-4 AutoEncoder == Method-2 AutoEncoder, No Need To Train Again.")
        print('-'*10) # Makes Output Readable
        print("Method-4 LSTM == Method-3 LSTM, No Need To Train Again.")
        print('-'*20) # Makes Output Readable


if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)