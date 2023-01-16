import pickle
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import optim
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F

from encoder.abstract_encoder import AbstractEncoder
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from part import Part

import matplotlib.pyplot as plt

USE_CUDA = True

device = "cuda" if USE_CUDA else "cpu"

# Known Id Range:
# Part id: 0 - 2270 (both inclusive)
# Family id: 0 - 100 (both inclusive)

class EncoderTrainingNetwork(nn.Module):

    def __init__(self, input_output_size: int):
        super(EncoderTrainingNetwork, self).__init__()

        layer2Size = 400
        encodingSize = 100

        self.fc1 = nn.Linear(input_output_size, layer2Size)
        self.fc2 = nn.Linear(layer2Size, encodingSize)
        self.fc3 = nn.Linear(encodingSize, layer2Size)
        self.fc4 = nn.Linear(layer2Size, input_output_size)

        self.lossCriterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.03)
        self.loss_per_step = []
        self.generation = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def training_step(self, data_x, data_y):
        self.optimizer.zero_grad()
        outputs = self(data_x)
        loss = self.lossCriterion(outputs, data_y)
        self.loss_per_step.append(float(loss))
        loss.backward()
        self.optimizer.step()
        self.generation += 1
        return loss


def constructTrainingData(rawData: List[Graph]) -> List[Tuple[Part, Part]]:
    """
    Constructs the data used to train the encoder. Returns a list of tuples, where each tuple is a pair
    of parts which are connected in a graph.
    """
    result = []
    for g in rawData:
        for (node, connectedNodes) in g.get_edges().items():
            for connectedNode in connectedNodes:
                result.append((node.get_part(), connectedNode.get_part()))
    return result


def loadTrainingData():
    with open('../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    return constructTrainingData(train_graphs)

def encodeTrainingData(rawTrainingData, encoder: AbstractEncoder):
    data_X = [encoder.encode(i[0]) for i in rawTrainingData]
    data_y = [encoder.encode(i[1]) for i in rawTrainingData]
    return (data_X, data_y)

def trainAutoEncoder(network: EncoderTrainingNetwork, X_train, y_train, batch_size = 20):

    training_step = 0

    for batch_start_index in range(0, len(X_train), batch_size):
        x = torch.from_numpy(np.array(X_train[batch_start_index:batch_start_index + batch_size])).to(device)
        y = torch.from_numpy(np.array(y_train[batch_start_index:batch_start_index + batch_size])).to(device)

        loss = network.training_step(x, y)
        training_step = training_step + 1
        print(f"\rExecuted training step {training_step} with loss {loss}.                    ", end="")
    print()
    return network

rawTrainingData = loadTrainingData()
print("Training data loaded.")
encoder = OneHotEncoder()
(data_X, data_y) = encodeTrainingData(rawTrainingData, encoder)
print("Training data encoded.")

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=encoder.get_encoding_size()).to(device)

for i in range(20):
    trainAutoEncoder(encoderTrainingNetwork, X_train, y_train, 150)

print("Training complete")

x = list(range(1, encoderTrainingNetwork.generation + 1))
y = encoderTrainingNetwork.loss_per_step
plt.plot(x, y)
plt.ylim(0, 0.002)
plt.show()