import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_config
from data_loader import ForexDataLoader
from dataset import TimeSeriesDataset
from models import TransformerModel, RNNModel


def train_lstm():
    config = load_config('./config.yml')
    file_name = config['file_name']

    fdl = ForexDataLoader()
    df = fdl.from_csv(file_name)
    train_test_batches = fdl.make_feature_target_sets(df, 
                                                      sequence_size = sequence_size, 
                                                      test_size = test_size,
                                                      batch_spacing = 5000)

    print('train_test_batches size: ', len(train_test_batches))
    train_df, test_df  = train_test_batches[batch_number]
    train_df_features = fdl.features(train_df)
    train_df_targets = fdl.targets(train_df)

    train_dataset = TimeSeriesDataset(train_df_features, train_df_targets, train_sequence_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    # Define hyperparameters

    input_size = train_df_features.shape[1]  # Size of input feature
    output_size = train_df_targets.shape[1] # Size of the target

    # model = TransformerModel(input_size, hidden_size, num_layers)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()

    model = RNNModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        total_loss = 0.0
        print('train_loader size: ', len(train_loader))
        cnt = 0
        for batch_x, batch_y in train_loader:
            cnt += 1
            print(f'batch # {cnt}/{len(train_loader)}')
            optimizer.zero_grad()
            # Forward pass: Compute predictions and loss
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'model.pth')

