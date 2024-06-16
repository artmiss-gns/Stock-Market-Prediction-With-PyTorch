from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class MakeSequence(Dataset):
    def __init__(self, data, sequence_length, target):
        self.target_column_number = np.where(data.columns == target)[0][0]
        self.target = target
        self.sequence_length = sequence_length
        self.data = data.values
        self.x, self.y = self._transform()

    def _transform(self):
        x = self._make_sequence()
        y = self._make_y(x)
        # Remove the last input sequence as it has no corresponding target
        x = x[:-1]
        return x, y

    def _make_sequence(self):
        sequenced_data = np.lib.stride_tricks.sliding_window_view(self.data, window_shape=(self.sequence_length, self.data.shape[1]))
        sequenced_data = np.squeeze(sequenced_data) # ! `squeeze` gets an argument `axis`, which should be set for higher dims
        return sequenced_data
    
    def _make_y(self, x):
        return x[1:, :, self.target_column_number]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float)
        return x, y