'''
Generate Results from Trained Models
'''

# Import Necessary Libraries
import platform
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
from PIL import Image
from torchvision import transforms
import glob
import shutil

# Import Model Definations
from autoencoder_model import Grey2RGBAutoEncoder
from lstm_model import ConvLSTM

# Define Universal Parameters
i = 3 # resolutions[i] to use in the Proejct as Image Size
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

# Define Backend for Distributed Computing
def get_backend():
    system_type = platform.system()
    if system_type == "Linux":
        return "nccl"
    else:
        return "gloo"

# Function to initialize the process group for distributed computing
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=get_backend(), rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Function to clean up the process group after computation
def cleanup():
    dist.destroy_process_group()

# The function to load your models
def load_model(model, model_path, device):
    map_location = lambda storage, loc: storage.cuda(device)
    state_dict = torch.load(model_path, map_location=map_location)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    # Move the model to the device and wrap the model with DDP after its state_dict has been loaded
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model

# Define the function to save images
def save_images(img_seq, img_dir, global_start_idx):
    to_pil = transforms.ToPILImage()
    for i, image_tensor in enumerate(img_seq):
        global_idx = global_start_idx + i  # Calculate the global index
        image = to_pil(image_tensor.cpu())
        image.save(f'{img_dir}/image_{global_idx:04d}.tif')

def reorder_and_save_images(img_exp_dir, output_dir):
    image_paths = glob.glob(os.path.join(img_exp_dir, 'image_*.tif'))
    sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    for i, img_path in enumerate(sorted_image_paths):
        img = Image.open(img_path)
        img.save(os.path.join(output_dir, f'enhanced_sequence_{i:04d}.tif'))

# The main function that will be executed by each process
def enhance(rank, world_size, img_inp_dir, img_exp_dir, lstm_path, autoencoder_path):
    setup(rank, world_size)
    lstm_model = ConvLSTM(input_dim=1, hidden_dims=[1, 1, 1], kernel_size=(3, 3), num_layers=3, alpha=0.6)
    lstm = load_model(lstm_model, lstm_path, rank)
    lstm.eval()
    autoencoder_model = Grey2RGBAutoEncoder()
    autoencoder = load_model(autoencoder_model, autoencoder_path, rank)
    autoencoder.eval()
    image_files = os.listdir(img_inp_dir)
    per_gpu = (len(image_files) + world_size - 1) // world_size
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, len(image_files))
    global_start_idx = start_idx
    local_images = [Image.open(os.path.join(img_inp_dir, image_files[i])) for i in range(start_idx, end_idx)]
    # Apply grayscale transformation only to the LSTM input tensors
    transform_lstm = transforms.Compose([
        transforms.Resize(resolutions[i]),
        transforms.Grayscale(),  # Convert the images to grayscale
        transforms.ToTensor(),
    ])
    local_tensors_lstm = torch.stack([transform_lstm(image) for image in local_images]).unsqueeze(0).to(rank)
    # Do not apply grayscale transformation to the other input tensors
    transform = transforms.Compose([
        transforms.Resize(resolutions[i]),
        transforms.ToTensor(),
    ])
    local_tensors = torch.stack([transform(image) for image in local_images]).unsqueeze(0).to(rank)
    with torch.no_grad():
        local_output_sequence, _ = lstm(local_tensors_lstm)
        local_output_sequence = local_output_sequence.squeeze(0)
    local_output_sequence = torch.cat([local_output_sequence, torch.zeros_like(local_output_sequence), torch.zeros_like(local_output_sequence)], dim=1)
    # Interleave the input and output images
    interleaved_sequence = torch.stack([t for pair in zip(local_tensors.squeeze(0), local_output_sequence) for t in pair])
    with torch.no_grad():
        local_output_enhanced = torch.stack([autoencoder(t.unsqueeze(0)) for t in interleaved_sequence]).squeeze(1)
    save_images(local_output_enhanced, img_exp_dir, global_start_idx)
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # Input Sequence Directory (All Methods)
    img_sequence_inp_dir = r'../Dataset/Inference/InputSequence'
    # Intermediate Results will be Stored in this Directory which later wll be re-ordered (All Methods)
    temp_dir = r'../Dataset/Inference/OutputSequence/Temp'
    os.makedirs(temp_dir, exist_ok=True)

    '''Working Directories for (Method-1)'''
    autoencoder_path = r'../Models/Method1/model_autoencoder_m1.pth'
    lstm_path = r'../Models/Method1/model_lstm_m1.pth'
    img_sequence_out_dir = r'../Dataset/Inference/OutputSequence/Method1/'
    os.makedirs(img_sequence_out_dir, exist_ok=True)

    '''Working Directories for (Method-2)'''
    # autoencoder_path = r'../Models/Method2/model_autoencoder_m2.pth'
    # lstm_path = r'../Models/Method1/model_lstm_m1.pth'
    # img_sequence_out_dir = r'../Dataset/Inference/OutputSequence/Method2/'
    # os.makedirs(img_sequence_out_dir, exist_ok=True)

    '''Working Directories for (Method-3)'''
    # autoencoder_path = r'../Models/Method1/model_autoencoder_m1.pth'
    # lstm_path = r'../Models/Method3/model_lstm_m3.pth'
    # img_sequence_out_dir = r'../Dataset/Inference/OutputSequence/Method3/'
    # os.makedirs(img_sequence_out_dir, exist_ok=True)

    '''Working Directories for (Method-4)'''
    # autoencoder_path = r'../Models/Method2/model_autoencoder_m2.pth'
    # lstm_path = r'../Models/Method3/model_lstm_m3.pth'
    # img_sequence_out_dir = r'../Dataset/Inference/OutputSequence/Method4/'
    # os.makedirs(img_sequence_out_dir, exist_ok=True)

    processes = []
    for rank in range(world_size):
        p = Process(target=enhance, args=(rank, world_size, img_sequence_inp_dir, temp_dir, lstm_path, autoencoder_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Reorder images once processing by all GPUs is complete
    reorder_and_save_images(temp_dir, img_sequence_out_dir)  
    # Delete all Intermediate Results
    shutil.rmtree(temp_dir)

