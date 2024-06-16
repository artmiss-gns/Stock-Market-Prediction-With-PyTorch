from torch import nn
import torch
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        """
        Initializes the LSTM module.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of LSTM layers.
        """
        super().__init__()
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.NUM_LAYERS = num_layers
        self.BATCH_SIZE = batch_size

        self.lstm = nn.LSTM(input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_SIZE, num_layers=self.NUM_LAYERS, batch_first=True)
        self.linear = nn.Linear(in_features=self.HIDDEN_SIZE, out_features=1)

    def get_hc(self, device='cpu'):
        h_0 = nn.Parameter(torch.zeros((self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_SIZE), dtype=torch.float).to(device))
        c_0 = nn.Parameter(torch.zeros((self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_SIZE), dtype=torch.float).to(device))
        
        return h_0, c_0
    
    def forward(self, x):
        """
        Performs a forward pass through the LSTM module.

        Args:
            x (torch.Tensor): The input data tensor of shape (batch_size, seq_len, input_size).

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                - output (torch.Tensor): The output tensor of shape (batch_size, 1).
                - hidden_states (tuple[torch.Tensor, torch.Tensor]): The final hidden and cell states of the LSTM.
        """
        h_0, c_0 = self.get_hc(x.device)

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.linear(output[:, -1, :])
        return output, (h_n, c_n)
