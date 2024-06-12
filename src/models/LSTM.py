from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, h_0, c_0):
        '''
            x : input X data 
            h_0 : hidden state (short-term memory)
            c_0 : cell state (long-term memory)
        '''
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.linear(output[:, -1, :])
        return output, (h_n, c_n)