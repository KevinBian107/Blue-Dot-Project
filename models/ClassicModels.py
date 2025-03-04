import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    '''Simple feedforward network for predicting pupil dilation'''
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, return_activations=False):
        act1 = torch.relu(self.fc1(x))
        act2 = torch.relu(self.fc2(act1))
        output = self.fc3(act2)

        if return_activations:
            return output, act1, act2
        return output
    
class LSTMModel(nn.Module):
    """Simple LSTM model for pupil dilation prediction with hidden state analysis"""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # h_n and c_n are the final hidden/cell states

        output = self.output_layer(lstm_out[:, -1, :])  # Take the last time step

        return output, h_n.squeeze(0), c_n.squeeze(0)  # Return hidden states
