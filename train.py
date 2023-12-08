import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from utils import load_config
from data_loader import ForexDataLoader
from models import Seq2SeqLSTM
from trainers import Seq2SeqTrainer
from sklearn.model_selection import train_test_split
from dataset import StackDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformer import Scaler
from forecasters import Seq2SeqForcaster

def train_seq2seg_lstm():
    config = load_config('./config.yml')
    data_csv_file_path = config['data_csv_file_name']
    sequence_length = config['sequence_length'] # lenght of the sequence 
    target_length = config['forecast_length']
    sequence_spacing = config['sequence_spacing']

    fdl = ForexDataLoader()
    df = fdl.from_csv(data_csv_file_path)
    print('Loading data... ')
    feature_target_sets = fdl.make_feature_target_sets(df, 
                                           sequence_length = sequence_length, 
                                           target_length = target_length,
                                           sequence_spacing = sequence_spacing)
    
    print('Train/Test split... ')
    test_size = 0.3
    train, test = train_test_split(feature_target_sets, 
                                             test_size=test_size, 
                                             shuffle=False)
    
    test_data_path = config['test_data_path']
    fdl.save(test, test_data_path)
    print(train[0][0].shape, train[0][1].shape)
    
    print('Normilizing... ')
    transformer = Scaler()

    # Create a DataLoader with data normalization for training
    batch_size = config['batch_size'] # size of input columns
    train_dataset = StackDataset(transformer.fit_transform(train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader without data normalization for evaluation
    test_dataset = StackDataset(transformer.transform(test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Store transformer
    scaler_function_path = config['scaler_function_path']
    joblib.dump(transformer, scaler_function_path)

    input_size = config['input_size'] # size of input columns
    hidden_size = config['hidden_size'] # Number of LSTM units
    num_layers = config['num_layers'] # Number of LSTM layers 

    output_size = config['output_size'] # size of output columns
    forecast_length = config['forecast_length'] # length of forecast seq

    model = Seq2SeqLSTM(input_size, hidden_size, num_layers, output_size, forecast_length)

    criterion = nn.MSELoss()  # Use Mean Squared Error as the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Seq2SeqTrainer(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    num_epochs = config['num_epochs']
    print('Training... ')
    trained_model = trainer.run(train_dataloader, num_epochs)
    model_path = config['model_path']
    trainer.save(output_path = model_path)

def train_batch_lstm():
    config = load_config('./config.yml')

    #  load data configs
    file_name = config['file_name']
    sequence_length = config['sequence_length'] # lenght of the sequence 
    target_length = config['forecast_length']
    batch_spacing = config['batch_spacing']

    # load data
    fdl = ForexDataLoader()
    df = fdl.from_csv(file_name)
    # df, scaler = fdl.normilize(df)
    feature_target_sets = fdl.make_feature_target_sets(df, 
                                           sequence_length = sequence_length, 
                                           target_length = target_length,
                                           batch_spacing = batch_spacing)

    print('data_batches size: ', len(data_batches))

    data_batches = fdl.make_batch_data(feature_target_sets, batch_size = 100)
    # Your array of (seq, target) tuples

    test_size = 0.2
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data_batches, test_size=test_size, shuffle=False)


    # load model config
    input_size = config['input_size'] # size of input columns
    hidden_size = config['hidden_size'] # Number of LSTM units
    output_size = config['output_size'] # size of output columns
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    sequence_length = config['sequence_length'] # length of the input sequence
    num_layers = config['num_layers']
    model_path = config['model_path']

    
    model = Seq2SeqRNN(input_size = input_size, 
                    hidden_size = hidden_size, 
                    num_layers = num_layers,
                    output_size = output_size,
                    prediction_length = target_length)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train = BatchTrain(model=model,
                    criterion=criterion,
                    optimizer=optimizer)

    for i, (x, y) in enumerate(train_data):
        # print(f"batch {i+1}/{len(train_data)}:")
        train_x = fdl.features(x)
        train_y = fdl.targets(y)
        print (train_x.shape, train_y.shape)
        if train_x.shape[1] != input_size:
            print(f"input_size does not match train_x shape {train_x.shape[1]} != {input_size}")
        if train_y.shape[1] != output_size:
            print(f"out_size does not match train_y shape {train_y.shape[1]} != {output_size}")
        loss = train.run(train_x, train_y, num_epochs=num_epochs)

    train.save(output_path = model_path)

def train_rolling_lstm():
    config = load_config('./config.yml')

    #  load data configs
    file_name = config['file_name']
    batch_size = config['batch_size'] # lenght of the sequence 
    train_batch_index = config['train_batch_index']

    # load data
    fdl = ForexDataLoader()
    df = fdl.from_csv(file_name)
    # df, scaler = fdl.normilize(df)
    data_batches = fdl.make_train_batchs(df, batch_size = batch_size, 
                                        batch_spacing = 5000)

    print('data_batches size: ', len(data_batches), data_batches.shape)
    train_data  = data_batches[train_batch_index]
    train_features = fdl.features(train_data)
    # train_y = fdl.targets(train_data)

    # load model config
    hidden_size = config['hidden_size']  # Number of LSTM units
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    predict_batch_index = config['predict_batch_index']
    sequence_length = config['sequence_length'] # length of the input sequence
    num_layers = config['num_layers']
    model_path = config['model_path']

    # Initialize the LSTM model
    model = LSTMModel(input_size = train_features.shape[1], 
                    hidden_size = hidden_size,
                    output_size = train_features.shape[1], 
                    num_layers = num_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train = RollingTrain(model=model,
                    criterion=criterion,
                    optimizer=optimizer)
    # Train the model
    print("training srarted...")
    loss = train.run(train_features, sequence_length = sequence_length, num_epochs=num_epochs)
    train.save(output_path = model_path)

    # Validate the model
    validate_batch_index = config['validate_batch_index']
    forecast_length = config['forecast_length']
    validate_data  = data_batches[validate_batch_index]
    validate_features = fdl.features(validate_data)
    forecasts, targets = train.validate(validate_features, forecast_length)
    print(forecasts.shape)
    print(targets.shape)


if __name__ == "__main__":
    # train_rolling_lstm()
    # train_batch_lstm()
    train_seq2seg_lstm()