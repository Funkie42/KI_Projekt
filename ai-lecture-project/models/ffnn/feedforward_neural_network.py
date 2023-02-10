import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import device

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, in_dimention, out_dimention, hidden_dimention_1, hidden_dimention_2, hidden_dimention_3):
        super(FeedforwardNeuralNetwork, self).__init__()

        # 4 FC Layers für das Training des Netzwerks
        self.fc1 = nn.Linear(in_dimention, hidden_dimention_1).to(device)
        self.fc2 = nn.Linear(hidden_dimention_1, hidden_dimention_2).to(device)
        self.fc3 = nn.Linear(hidden_dimention_2, hidden_dimention_3).to(device)
        self.fc4 = nn.Linear(hidden_dimention_3, out_dimention).to(device)

        # Normieren des Outputs mit Tanh
        self.sigmoid = nn.Sigmoid()

        # Init der Gewichte mit He Initiation
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')

    def forward(self, input):
        # Output ist das, was durch die FC Layers geschleust wird und am Ende das Ergebnis ist
        output = self.fc1(input)
        # Relu nach jedem FC Layer um die Werte zu normieren (<0 verhindern)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        output = F.relu(output)
        output = self.fc4(output)
        output = self.sigmoid(output)
        return output

