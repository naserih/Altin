import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class Scaler():
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()


    def fit_transform(self, data):
        features = [sample[0].numpy() for sample in data]
        targets = [sample[1].numpy() for sample in data]
        all_features = np.vstack(features)
        all_targets = np.vstack(targets)
        print('all_targets and all_targets: ', all_features.shape, all_targets.shape)
        self.feature_scaler.fit(all_features)
        self.target_scaler.fit(all_targets)

        scaled_data = [(torch.tensor(self.feature_scaler.transform(f), dtype=torch.float32),
                        torch.tensor(self.target_scaler.transform(t), dtype=torch.float32))
                    for f, t in data]
        return scaled_data

    def transform(self, data):
        scaled_data = [(torch.tensor(self.feature_scaler.transform(f), dtype=torch.float32),
                        torch.tensor(self.target_scaler.transform(t), dtype=torch.float32))
                    for f, t in data]
        return scaled_data

    def target_inverse_transform(self, scaled_data):
        reshaped_date = scaled_data.reshape(-1, scaled_data.shape[-1])
        inverted_data = np.array(self.target_scaler.inverse_transform(reshaped_date))
        inverted_data = inverted_data.reshape(scaled_data.shape)
        inverted_data = torch.tensor(inverted_data)
        return inverted_data

    def feature_inverse_transform(self, scaled_data):
        reshaped_date = scaled_data.reshape(-1, scaled_data.shape[-1])
        inverted_data = np.array(self.feature_scaler.inverse_transform(reshaped_date))
        inverted_data = inverted_data.reshape(scaled_data.shape)
        inverted_data = torch.tensor(inverted_data)
        return inverted_data
