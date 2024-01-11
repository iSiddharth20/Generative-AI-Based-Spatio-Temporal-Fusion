'''
Generate Results from Trained Models
'''

# Import Necessary Libraries
from PIL import Image
import torch
from torchvision import transforms
import os

# Define Universal Variables
image_width = 1280
image_height = 720

# Define the Transformation
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.Grayscale(),  # Convert the images to grayscale
    transforms.ToTensor(),
])

def save_images(img_seq, img_dir):
    # Define the transformation to convert the tensors to PIL images
    to_pil = transforms.ToPILImage()
    # Iterate over the tensors in out_seq_enhanced
    for i, image_tensor in enumerate(img_seq):
        # Convert the tensor to a PIL image
        image = to_pil(image_tensor)
        # Save the image
        image.save(f'{img_dir}/image_{i}.tif')

def EnhanceSequence(model_lstm, model_autoencoder, img_inp_dir, export_seq=False, img_exp_dir=None):
    # Load Images and Transform for Generation
    image_files = os.listdir(img_inp_dir)
    images = [Image.open(os.path.join(img_inp_dir, image_file)) for image_file in image_files]
    inputs = torch.stack([transform(image) for image in images])
    # Pass the Input Image Sequence to LSTM Model
    out_seq , _ = model_lstm(inputs)
    # Initialize an empty list to store the resulting sequence
    resulting_sequence = []
    # Iterate over the images in inputs and out_seq
    for input_image, generated_image in zip(inputs, out_seq):
        # Add the input image to the resulting sequence
        resulting_sequence.append(input_image)
        # Add the generated image to the resulting sequence
        resulting_sequence.append(generated_image)
    # If inputs has one more image than out_seq, add the last image from inputs to the resulting sequence
    if len(inputs) > len(out_seq):
        resulting_sequence.append(inputs[-1])
    # Convert the list of images in the resulting sequence to a PyTorch tensor
    resulting_sequence = torch.stack(resulting_sequence)
    # Initialize an empty list to store the enhanced images
    out_seq_enhanced = []
    # Iterate over the images in the resulting sequence
    for image in resulting_sequence:
        # Add an extra dimension to the image tensor and pass it through the AutoEncoder model
        enhanced_image = model_autoencoder(image.unsqueeze(0))
        # Remove the extra dimension from the enhanced image and add it to out_seq_enhanced
        out_seq_enhanced.append(enhanced_image.squeeze(0))
    # Convert the list of enhanced images to a PyTorch tensor
    out_seq_enhanced = torch.stack(out_seq_enhanced)
    if export_seq:
        save_images(out_seq_enhanced, img_exp_dir)

# Define Working Directories (Create Output Directories if they Don't Exist)
img_sequence_inp_dir = '../Dataset/Interence/InputSequence/'
img_sequence_out_dir_m1 = '../Dataset/Interence/OutputSequence/Method1/'
os.makedirs(img_sequence_out_dir_m1, exist_ok=True) # Creating Directory for Output Sequence 
img_sequence_out_dir_m2 = '../Dataset/Interence/OutputSequence/Method2/'
os.makedirs(img_sequence_out_dir_m2, exist_ok=True) # Creating Directory for Output Sequence 
img_sequence_out_dir_m3 = '../Dataset/Interence/OutputSequence/Method3/'
os.makedirs(img_sequence_out_dir_m3, exist_ok=True) # Creating Directory for Output Sequence 
img_sequence_out_dir_m4 = '../Dataset/Interence/OutputSequence/Method4/'
os.makedirs(img_sequence_out_dir_m4, exist_ok=True) # Creating Directory for Output Sequence 
print('Working Directories Defined')
print('-'*20) # Makes Output Readable

# Load and Set the AutoEncoder Models to evaluation mode
try:
    model_autoencoder_m1 = torch.load('../Models/Method1/model_autoencoder_m1.pth')
    model_autoencoder_m1.eval()
    print('Method 1 AutoEncoder Model Loaded')
    print('-'*10) # Makes Output Readable
except:
    print('Method 1 AutoEncoder Model Not Found')
    print('-'*20)
try:
    model_autoencoder_m2 = torch.load('../Models/Method2/model_autoencoder_m2.pth')
    model_autoencoder_m2.eval()
    print('Method 2 AutoEncoder Model Loaded')
    print('-'*10) # Makes Output Readable
except:
    print('Method 2 AutoEncoder Model Not Found')
    print('-'*20)

# Load and Set the LSTM Models to evaluation mode
try:
    model_lstm_m1 = torch.load('../Models/Method1/model_lstm_m1.pth')
    model_lstm_m1.eval()  # Set the model to evaluation mode
    print('Method 1 LSTM Model Loaded')
    print('-'*10) # Makes Output Readable
except:
    print('Method 1 LSTM Model Not Found')
    print('-'*20)
try:
    model_lstm_m3 = torch.load('../Models/Method3/model_lstm_m3.pth')
    model_lstm_m3.eval()  # Set the model to evaluation mode
    print('Method 3 LSTM Model Loaded')
    print('-'*10) # Makes Output Readable
except:
    print('Method 3 LSTM Model Not Found')
    print('-'*20)

EnhanceSequence(model_lstm_m1, model_autoencoder_m1, img_sequence_inp_dir, export_seq=True, img_exp_dir=img_sequence_out_dir_m1)
EnhanceSequence(model_lstm_m1, model_autoencoder_m2, img_sequence_inp_dir, export_seq=True, img_exp_dir=img_sequence_out_dir_m2)
EnhanceSequence(model_lstm_m3, model_autoencoder_m1, img_sequence_inp_dir, export_seq=True, img_exp_dir=img_sequence_out_dir_m3)
EnhanceSequence(model_lstm_m3, model_autoencoder_m2, img_sequence_inp_dir, export_seq=True, img_exp_dir=img_sequence_out_dir_m4)