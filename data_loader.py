import torch
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle

class ForexDataLoader():

    @staticmethod
    def from_csv(file_name):
        df = pd.read_csv(file_name, index_col=1,parse_dates=True)
        # df['time'] = pd.to_datetime(df['time'])
        df.index.name = 'Date'
        return df
    
    def normilize(self, df):
        columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
        scaler = MinMaxScaler()
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df, scaler


    def make_feature_target_sets(self, df, 
                               sequence_length = 10000, 
                               target_length = 100,
                               sequence_spacing = 5000):
        feature_target_sets = []

        # Random gap between two batches (samples i.e. min)
        # sequence_spacing: set 0 to sample all points

        seg_start = 0
        while seg_start + target_length <= len(df):
            training_end = seg_start + sequence_length
            target_start = training_end
            target_end = target_start + target_length
            training_data = df.iloc[seg_start:training_end]
            target_data = df.iloc[target_start:target_end]
            if len(training_data) != 0 and len(target_data) != 0:
                feature_target_sets.append((self.features(training_data),
                                            self.targets(target_data)))
            #  Random gap between two batches (hrs)
            random_gap = random.randint(target_length, target_length + sequence_spacing)
            seg_start += random_gap

        return feature_target_sets


    def make_batch_data(self, feature_target_sets, 
                            batch_size = 100):
        features_batchs = [] 
        target_batchs = []
        for i in range(0, len(feature_target_sets), batch_size):
            batch = feature_target_sets[i:i+batch_size]
            features_batch = torch.stack([self.features(item[0]) for item in batch], dim=0)
            target_batch = torch.stack([self.targets(item[1]) for item in batch], dim=0)
            features_batchs.append(features_batch)
            target_batchs.append(target_batch)

        return features_batchs, target_batchs




        return feature_target_sets
    
    def make_train_batchs(self, df, 
                               batch_size = 10000, 
                               batch_spacing = 5000):
        train_epochs = []

        # Random gap between two batches (samples i.e. min)
        # batch_spacing: set 0 to sample all points

        seg_start = 0
        while seg_start <= len(df):
            training_end = seg_start + batch_size
            training_data = df.iloc[seg_start:training_end]
            train_epochs.append(training_data)
            #  Random gap between two batches (hrs)
            random_gap = random.randint(0, batch_spacing)
            seg_start += random_gap

        return train_epochs
    
    def features(self, df):
        df_features = df[[
            'timestamp', 
            'open', 'high', 'low', 'close', 
            'volume', 'vwap', 'transactions'
            ]]
        return torch.tensor(df_features.to_numpy(), dtype=torch.float32)

    def targets(self, df):
        df_targets = df[['close']]
        return torch.tensor(df_targets.to_numpy(), dtype=torch.float32)

    @staticmethod
    def save(data, dest):
        with open(dest, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load(dest):
        with open(dest, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

        

