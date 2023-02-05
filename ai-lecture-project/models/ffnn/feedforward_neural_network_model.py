import torch
import torch.nn as nn

from config.config import device

class FeedforwardNeuralNetworkModel(nn.Module):
    def __init__(self, in_dimention, out_dimention, hidden_dimention):
        super(FeedforwardNeuralNetworkModel, self).__init__()

        # 1. FC Layer
        self.fc1 = nn.Linear(in_dimention, hidden_dimention).to(device)
        # Normieren
        self.tanh = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dimention, out_dimention).to(device)

    def forward(self, input):
        # Output ist das, was mal output sein wird
        output = self.fc1(input)
        output = self.tanh(output)
        output = self.fc2(output)
        return output

