from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class MakeSequence(Dataset):
    def __init__(self, data, sequence_length, target):
        self.target_column_number = np.where(data.columns == target)[0][0]
        self.target = target # target feature
        self.sequence_length = sequence_length
        self.data = data.values # convert it to numpy array
        self.x, self.y = self._transform()
        self.x = torch.tensor(self.x, dtype=torch.float)
        self.y = torch.tensor(self.y, dtype=torch.float)

    def _transform(self):
        x = self._make_sequence()
        y = self._make_y(x)
        return x, y

    def _make_sequence(self):
        # sequenced_data = np.lib.stride_tricks.sliding_window_view(self.data, (self.sequence_length))
        sequenced_data = np.lib.stride_tricks.sliding_window_view(self.data, window_shape=(self.sequence_length, self.data.shape[1]))
        sequenced_data = np.squeeze(sequenced_data) # ! `squeeze` gets an argument `axis`, which should be set for higher dims
        return sequenced_data
    
    def _make_y(self, x):
        return x[1:, :, self.target_column_number]
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index, :], self.y[index]