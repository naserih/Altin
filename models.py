import torch
import torch.nn as nn



# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Consider only the last time step
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(input_size, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x, tgt):
        x = self.transformer(x)
        x = self.decoder(x)
        return x
        
    def predict(self, x):
        # The input src should have shape (sequence_length, batch_size, input_size)
        out = self.transformer(x)
        # out = out[-1, :, :]  # Consider only the last time step in the sequence
        out = self.decoder(out)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The input x should have shape (batch_size, sequence_length, input_size)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Consider only the last time step in the sequence
        return out
    
    def predict(self, x):
        self.eval()
        # Ensure the input has the right dimensions (sequence_length, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add a batch dimension

        # Reshape input_data to (1, sequence_length, input_size)
        x = x.unsqueeze(0)

        # Pass the input data through the model
        with torch.no_grad():
            predictions = self(x)
            # Remove the batch dimension from predictions
        predictions = predictions.squeeze(0)
        return predictions

class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_length):
        super(Seq2SeqRNN, self).__init__()
        self.prediction_length = prediction_length
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _, hidden_state = self.encoder(x)
        output, _ = self.decoder(x, hidden_state)
        output = self.fc(output[:, -self.prediction_length:, :])
        return output


class SegCNNModel(nn.Module):
    def __init__(self):
        super(SegCNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2),
            nn.Sigmoid()  # You can adjust this activation function as needed
        )

    def forward(self, x):
        print('1 > ', x.shape)
        x = self.encoder(x)
        print('2 > ', x.shape)
        x = self.decoder(x)
        return x

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_length):
        super(Seq2SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_length = forecast_length

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)  # Linear layer to match output size

    def forward(self, input_seq):
        encoder_output, (hidden, cell) = self.encoder(input_seq)
        decoder_input = hidden
        output_seq = torch.zeros( self.forecast_length, input_seq.shape[1], self.output_size)  # forecast_length x output size
        for t in range(self.forecast_length):  # forecast_length
            decoder_output, (hidden, cell) = self.decoder(decoder_input)
            output = self.linear(decoder_output)
            output_seq[t, :, :] = output.squeeze(0)  # Store the output at this time step
            decoder_input = hidden

        return output_seq
