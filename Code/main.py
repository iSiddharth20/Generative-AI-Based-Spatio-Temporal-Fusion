'''
Main Module
--------------------------------------------------------------------------------
'''

from data import Dataset
from lstm_model import LSTMModel
from autoencoder_model import Autoencoder
from losses import LossMLP, LossMEP
from training import Trainer
from evaluation import Evaluator

def main():
    # Prepare the dataset
    dataset = Dataset(image_size=(400, 600), batch_size=64, augment=True)
    (train_gray, val_gray, test_gray), (train_rgb, val_rgb, test_rgb) = dataset.split_data()

    # Convert data to PyTorch tensor dataloaders
    train_loader_gray = dataset.get_batches(train_gray)
    val_loader_gray = dataset.get_batches(val_gray)
    test_loader_gray = dataset.get_batches(test_gray)

    train_loader_rgb = dataset.get_batches(train_rgb)
    val_loader_rgb = dataset.get_batches(val_rgb)
    test_loader_rgb = dataset.get_batches(test_rgb)

    # Parameter setup (these values should be carefully chosen or tuned)
    input_feature_size = 1  # For grayscale there is only one input feature per pixel
    hidden_size = 128  # Number of features in the hidden state of the LSTM
    sequence_length = 400 * 600  # The number of features in the input sequence
    lstm_output_size = 400 * 600  # The output is a sequence of pixel values
    autoencoder_channels = 3  # For RGB images, the channel depth is 3

    # Initialize Models
    lstm_model = LSTMModel(input_feature_size, hidden_size, sequence_length, lstm_output_size)
    autoencoder_model = Autoencoder(image_channels=autoencoder_channels)

    # Initialize Loss Functions
    loss_mlp = LossMLP(alpha=0.5)
    loss_mep = LossMEP(alpha=0.5)

    '''
    Train and Evaluate with MLP
    '''
    # Initialize Trainer
    trainer_lstm = Trainer(lstm_model, loss_mlp)
    trainer_autoencoder = Trainer(autoencoder_model, loss_mlp)
    # Train Model
    trainer_lstm.train(100, train_loader_gray, val_loader_gray)
    trainer_autoencoder.train(100, train_loader_rgb, val_loader_rgb)
    # Initialize Evaluator
    evaluator_lstm = Evaluator(lstm_model, loss_mlp)
    evaluator_autoencoder = Evaluator(autoencoder_model, loss_mlp)
    # Evaluate Model
    evaluator_lstm.evaluate(test_loader_gray)
    evaluator_autoencoder.evaluate(test_loader_rgb)

    '''
    Train and Evaluate with MEP
    '''
    # Initialize Trainer
    trainer_lstm = Trainer(lstm_model, loss_mep)
    trainer_autoencoder = Trainer(autoencoder_model, loss_mep)
    # Train Model
    trainer_lstm.train(100, train_loader_gray, val_loader_gray)
    trainer_autoencoder.train(100, train_loader_rgb, val_loader_rgb)
    # Initialize Evaluator
    evaluator_lstm = Evaluator(lstm_model, loss_mep)
    evaluator_autoencoder = Evaluator(autoencoder_model, loss_mep)
    # Evaluate Model
    evaluator_lstm.evaluate(test_loader_gray)
    evaluator_autoencoder.evaluate(test_loader_rgb)

if __name__ == '__main__':
    main()