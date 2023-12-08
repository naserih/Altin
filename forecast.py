import joblib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import load_config
from data_loader import ForexDataLoader
from models import Seq2SeqLSTM
from dataset import StackDataset
from forecasters import Seq2SeqForcaster
from trainers import load_trained_model


config = load_config('./config.yml')

# Load test data
print('Loading data... ')
test_data_path = config['test_data_path']
test = ForexDataLoader.load(test_data_path)

# load transformer
scaler_function_path = config['scaler_function_path']
transformer = joblib.load(scaler_function_path)

# Create a DataLoader
batch_size = config['batch_size'] # size of input columns
test_dataset = StackDataset(transformer.transform(test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load model
trained_model_path = config['model_path']
trained_model, checkpoint = load_trained_model(Seq2SeqLSTM, trained_model_path)

# forecast test data
forecaster = Seq2SeqForcaster(trained_model)
test_inputs, test_targets, test_forecasts = forecaster.run(test_dataloader)
print(test_inputs.shape, test_targets.shape, test_forecasts.shape)
reverted_inputs = transformer.feature_inverse_transform(test_inputs)
reverted_targets = transformer.target_inverse_transform(test_targets)
reverted_forecasts = transformer.target_inverse_transform(test_forecasts)

# print(reverted_inputs.shape, reverted_forecasts.shape, test_targets.shape)

num_samples, input_seq_len, _ = test_inputs.shape
forcast_seq_len = test_targets.shape[1]

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

i = 500
print(test_inputs[i, :, 0].numpy())
axs[0].plot(range(input_seq_len), reverted_inputs[i, :, 4].numpy(), label=f'Input')
axs[0].plot(range(input_seq_len, input_seq_len + forcast_seq_len), reverted_targets[i, :, 0].numpy(), label=f'Target')
axs[1].plot(range(input_seq_len, input_seq_len + forcast_seq_len), reverted_targets[i, :, 0].numpy(), label=f'Target')
axs[1].plot(range(input_seq_len, input_seq_len + forcast_seq_len), reverted_forecasts[i, :, 0].numpy(), label=f'Forecast')

# Add labels and title
axs[0].legend()
axs[1].legend()
plt.xlabel('Time Step')
plt.suptitle('Line Plots for Each Data Point')

# Display the plots
plt.show()
