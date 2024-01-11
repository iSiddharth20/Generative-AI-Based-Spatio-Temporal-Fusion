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
import platform
import time

# Define Working Directories
autoencoder_grayscale_dir = '../Dataset/AutoEncoder/Grayscale'
autoencoder_rgb_dir = '../Dataset/AutoEncoder/RGB'
lstm_gray_sequences_dir = '../Dataset/LSTM'

# Define Universal Parameters
i = 2 # resolutions[i] to use in the Proejct as Image Size
resolutions = [
    (270, 480),
    (360, 640),
    (480, 854),
    (540, 960),
    (720, 1280),
    (900, 1600),
    (1080, 1920),
    (1440, 2560)
]

def get_backend():
    system_type = platform.system()
    if system_type == "Linux":
        return "nccl"
    else:
        return "gloo"

def main_worker(rank, world_size):
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    # Initialize the distributed environment.
    torch.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group(backend=get_backend(), init_method="env://", world_size=world_size, rank=rank)
    main(rank)  # Call the existing main function.

def main(rank):
    # Initialize Dataset Object (PyTorch Tensors)
    try:
        dataset_ae = CustomDataset(autoencoder_grayscale_dir, autoencoder_rgb_dir, lstm_gray_sequences_dir, resolutions[i])
        dataset_lstm = CustomDataset(autoencoder_grayscale_dir, autoencoder_rgb_dir, lstm_gray_sequences_dir, resolutions[i], for_lstm=True)
        if rank == 0:
            print('-'*20) # Makes Output Readable
            print('Importing Dataset Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Importing Dataset In-Complete : \n{e}")
    if rank == 0:
        print('-'*20) # Makes Output Readable
    # Import Loss Functions
    try:
        loss_mse = LossMSE() # Mean Squared Error Loss
        loss_mep = LossMEP(alpha=0.15) # Maximum Entropy Loss
        loss_ssim = SSIMLoss(alpha=0.1) # Structural Similarity Index Measure Loss
        if rank == 0:
            print('Importing Loss Functions Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Importing Loss Functions In-Complete : \n{e}")
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Initialize AutoEncoder Model and Import Dataloader (Training, Validation)
    data_autoencoder_train, data_autoencoder_val = dataset_ae.get_autoencoder_batches(val_split=0.25, batch_size=32)
    if rank == 0:
        print('AutoEncoder Model Data Imported.')
    model_autoencoder = Grey2RGBAutoEncoder()
    if rank == 0:
        print('AutoEncoder Model Initialized.')
        print('-'*20) # Makes Output Readable

    # Initialize LSTM Model and Import Dataloader (Training, Validation)
    data_lstm_train, data_lstm_val = dataset_lstm.get_lstm_batches(val_split=0.2, sequence_length=30, batch_size=12)
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
    optimizer = torch.optim.Adam(model_autoencoder.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
    trainer_autoencoder_baseline = Trainer(model=model_autoencoder, 
                                           loss_function=loss_mse, 
                                           optimizer=optimizer,
                                           lr_scheduler=lr_scheduler, 
                                           model_save_path=model_save_path_ae, 
                                           rank=rank)
    if rank == 0:
        print('Method-1 AutoEncoder Trainer Initialized.')
    model_save_path_lstm = '../Models/Method1/model_lstm_m1.pth'
    optimizer = torch.optim.SGD(model_lstm.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    trainer_lstm_baseline = Trainer(model=model_lstm, 
                                    loss_function=loss_mse, 
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler, 
                                    model_save_path=model_save_path_lstm, 
                                    rank=rank)
    if rank == 0:
        print('Method-1 LSTM Trainer Initialized.')
        print('-'*10) # Makes Output Readable

    # Method 2 : Composite Loss (MSE + MaxEnt) for AutoEncoder and Mean Squared Error Loss for LSTM
    os.makedirs('../Models/Method2', exist_ok=True) # Creating Directory for Model Saving
    model_save_path_ae = '../Models/Method2/model_autoencoder_m2.pth'
    optimizer = torch.optim.Adam(model_autoencoder.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
    trainer_autoencoder_m2 = Trainer(model=model_autoencoder, 
                                     loss_function=loss_mep, 
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler, 
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
    optimizer = torch.optim.SGD(model_lstm.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    trainer_lstm_m3 = Trainer(model=model_lstm, 
                              loss_function=loss_ssim, 
                              optimizer=optimizer,
                              lr_scheduler=lr_scheduler,  
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
        epochs = 100
        if rank == 0:
            print('Method-1 AutoEncoder Training Start')
            start_time = time.time()
        stats_autoencoder_m1 = trainer_autoencoder_baseline.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        if rank == 0:
            print('Method-1 AutoEncoder Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-1 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")
            trainer_autoencoder_baseline.cleanup_ddp()
    if rank == 0:
        print('-'*10) # Makes Output Readable
    try:
        epochs = 100
        if rank == 0:
            print('Method-1 LSTM Training Start')
            start_time = time.time()
        stats_lstm_m1 = trainer_lstm_baseline.train_lstm(epochs, data_lstm_train, data_lstm_val)
        if rank == 0:
            print('Method-1 LSTM Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-1 LSTM Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")
            trainer_lstm_baseline.cleanup_ddp()
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Method-2
    try:
        epochs = 100
        if rank == 0:
            print('Method-2 AutoEncoder Training Start')
            start_time = time.time()
        stats_autoencoder_m2 = trainer_autoencoder_m2.train_autoencoder(epochs, data_autoencoder_train, data_autoencoder_val)
        if rank == 0:
            print('Method-2 AutoEncoder Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-2 AutoEncoder Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")
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
        epochs = 100
        if rank == 0:
            print('Method-3 LSTM Training Start.')
            start_time = time.time()
        stats_lstm_m3 = trainer_lstm_m3.train_lstm(epochs, data_lstm_train, data_lstm_val)
        if rank == 0:
            print('Method-3 LSTM Training Complete.')
    except Exception as e:
        if rank == 0:
            print(f"Method-3 LSTM Training Error : \n{e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")
            trainer_lstm_m3.cleanup_ddp()
    if rank == 0:
        print('-'*20) # Makes Output Readable

    # Method-4
    if rank == 0:
        print("Method-4 AutoEncoder == Method-2 AutoEncoder, No Need To Train Again.")
        print('-'*10) # Makes Output Readable
        print("Method-4 LSTM == Method-3 LSTM, No Need To Train Again.")
        print('-'*20) # Makes Output Readable

    # Print Stats of Each Model 
    if rank == 0:
        print('Best Stats for Baseline AutoEncoder :')
        epoch_num, train_loss, val_loss = stats_autoencoder_m1
        print(f'\tEpoch: {epoch_num} --- Training Loss: {train_loss} --- Validation Loss: {val_loss}')
        print('-'*20) # Makes Output Readable
        print('Best Stats for Baseline LSTM :')
        epoch_num, train_loss, val_loss = stats_lstm_m1
        print(f'\tEpoch: {epoch_num} --- Training Loss: {train_loss} --- Validation Loss: {val_loss}')
        print('-'*20) # Makes Output Readable
        print('Best Stats for Method-2 AutoEncoder :')
        epoch_num, train_loss, val_loss = stats_autoencoder_m2
        print(f'\tEpoch: {epoch_num} --- Training Loss: {train_loss} --- Validation Loss: {val_loss}')
        print('-'*20) # Makes Output Readable
        print('Best Stats for Method-3 LSTM :')
        epoch_num, train_loss, val_loss = stats_lstm_m3
        print(f'\tEpoch: {epoch_num} --- Training Loss: {train_loss} --- Validation Loss: {val_loss}')
        print('-'*20) # Makes Output Readable


if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)