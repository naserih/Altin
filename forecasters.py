import torch

class Seq2SeqForcaster():
    def __init__(self, model):
        self.model = model

    def run(self, dataloader):
        batch_inputs = []
        batch_predictions = []
        batch_targets = []
        for batch in dataloader:
            inputs, targets = batch
            batch_inputs.append(inputs)
            # Ensure the input sequence is in the correct format (sequence length, batch size, input size)
            inputs = inputs.permute(1, 0, 2)
            # print('inputs.shape: ', inputs.shape,  targets.shape)
            with torch.no_grad():
                outputs = self.model(inputs)
            outputs = outputs.permute(1, 0, 2)
            batch_predictions.append(outputs)
            batch_targets.append(targets)

        batch_inputs = torch.cat(batch_inputs, dim=0)
        batch_predictions = torch.cat(batch_predictions, dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)

        return batch_inputs, batch_targets, batch_predictions
        

class RollingPredict():
    def __init__(self, model, checkpoint):
        self.model = model
        self.sequence_length = checkpoint['sequence_length']
    def run(self, data, forecast_length=None):

        if forecast_length == None:
            forecast_length = 1
        targets = data[len(data)-forecast_length:]
        start_index = len(data) - self.sequence_length - forecast_length
        input_seq = data[start_index : start_index + self.sequence_length]
        forecasts = []
        input_seq = data[start_index : start_index + self.sequence_length]
        for i in range(forecast_length):
            inputs = input_seq.unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(inputs)
                # print(prediction.squeeze())
                forecasts.append(prediction)
            input_seq = torch.cat((input_seq[1:], prediction), dim=0)
        return torch.cat(forecasts), targets
