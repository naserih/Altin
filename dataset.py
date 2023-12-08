import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        x = torch.tensor(self.features[idx:end_idx], dtype=torch.float32)
        y = torch.tensor(self.targets[end_idx], dtype=torch.float32)
        return x, y

class StackDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_seq, target_seq = self.data[index]
        return input_seq, target_seq

    
