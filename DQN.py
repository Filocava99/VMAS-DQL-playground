import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import Device


class DQN(nn.Module):
    pass


class SimpleSequentialDQN(DQN):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(Device.get())

    def forward(self, x):
        return self.net(x)

class ComplexSequentialDQN(DQN):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(Device.get())

    def forward(self, x):
        return self.net(x)


class FusedLSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FusedLSTMPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(input_sequence)

        # Extract the output at the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through the fully connected layer
        output = self.fc(last_output)
        return output

class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)