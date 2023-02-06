import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import device

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, in_dimention, out_dimention, hidden_dimention_1, hidden_dimention_2, hidden_dimention_3):
        super(FeedforwardNeuralNetwork, self).__init__()
        # Normieren
        self.tanh = nn.Tanh()

        # 1. FC Layer
        self.fc1 = nn.Linear(in_dimention, hidden_dimention_1).to(device)
        self.fc2 = nn.Linear(hidden_dimention_1, hidden_dimention_2).to(device)
        self.fc3 = nn.Linear(hidden_dimention_2, hidden_dimention_3).to(device)
        self.fc4 = nn.Linear(hidden_dimention_3, out_dimention).to(device)

        self.sigmoid = nn.Sigmoid()

        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')

    def forward(self, input):
        # Output ist das, was mal output sein wird
        output = self.fc1(input)
        #output = F.relu(output)
        output = self.fc2(output)
        output = self.fc3(output)
        #output = F.relu(output)
        output = self.fc4(output)
        output = self.sigmoid(output)
        return output

