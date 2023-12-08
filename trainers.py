import torch

class Seq2SeqTrainer():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def run(self, train_dataloader, num_epochs = 100):
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                # Ensure the input sequence is in the correct format (sequence length, batch size, input size)
                inputs = inputs.permute(1, 0, 2)
                targets = targets.permute(1, 0, 2)
                # print('inputs.shape: ', inputs.shape,  targets.shape)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        return self.model
    
    def save(self, output_path='./model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'output_size': self.model.output_size,
            'num_layers': self.model.num_layers,
            'forecast_length': self.model.forecast_length,
        }, output_path)
        print(f'model saved to: {output_path}')

class BatchTrain():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.sequence_length = None

    def run(self, features, targets, num_epochs = 100):
        inputs = features.unsqueeze(0)
        targets = targets.unsqueeze(0)

        print('inputs.shape: ', inputs.shape, targets.shape)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        # print(f" \t epoch {epoch+1}/{num_epochs}: Loss: {loss.item():.4f}")
        return loss

class RollingTrain():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.sequence_length = None

    def run(self, train_features, sequence_length=None, num_epochs = 100):
        if sequence_length == None:
            sequence_length = len(train_features) - 1
        
        self.sequence_length = sequence_length

        for epoch in range(num_epochs):
            for i in range(len(train_features) - sequence_length):
                inputs = train_features[i: i + sequence_length].unsqueeze(0)
                targets = train_features[i + sequence_length: i + sequence_length + 1]  # Predict the next point (n+1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            # print(f"epoch {epoch+1}/{num_epochs}: Loss: {loss.item():.4f}")
        return loss
    
    def save(self, output_path='./model.pth'):
        if self.sequence_length == None:
            print("Error: model is not trained!")
            return None

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'output_size': self.model.output_size,
            'num_layers': self.model.num_layers,
            'sequence_length': self.sequence_length,
        }, output_path)
        print(f'model saved to: {output_path}')
    
    def validate(self, validate_features, forecast_length):
        forcast_start = len(validate_features) - forecast_length
        inputs = validate_features[forcast_start-self.sequence_length:forcast_start].unsqueeze(0)
        targets = validate_features[forcast_start: forcast_start + 1]
        print(inputs.shape)
        print(targets.shape)
        # self.model()
        forecasts = []
        for i in range(forecast_length):
            with torch.no_grad():
                prediction = self.model(inputs)
                forecasts.append(prediction.squeeze())
        return(torch.cat(forecasts), targets)

def load_trained_model(model, model_path):
    # Load the model and architectural details
    checkpoint = torch.load(model_path)
    loaded_model = model(
        input_size = checkpoint['input_size'],
        hidden_size = checkpoint['hidden_size'],
        output_size = checkpoint['output_size'],
        num_layers = checkpoint['num_layers'],
        forecast_length = checkpoint['forecast_length'],
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    print(f'model loaded from: {model_path}')
    return loaded_model, checkpoint

