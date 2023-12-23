# Import Necessary Libraries
import os
from PIL import Image
import torch

# Function that generates RGB Image Sequence with Interpolated Frames from a Grayscale Image Sequence 
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