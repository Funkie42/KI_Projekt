import pickle
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import optim, Tensor
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F

from encoder.abstract_encoder import AbstractEncoder
from encoder.binary_encoder import BinaryEncoder
from encoder.one_hot_encoder import OneHotEncoder
from config.config import device
from graph import Graph
from part import Part

import matplotlib.pyplot as plt


# Known Id Range:
# Part id: 0 - 2270 (both inclusive)
# Family id: 0 - 100 (both inclusive)

class EncoderTrainingNetwork(nn.Module):

    def __init__(self, input_output_size: int):
        super(EncoderTrainingNetwork, self).__init__()

        layer_2_size = 400
        encodingSize = 100

        self.fc1 = nn.Linear(input_output_size, layer_2_size)
        self.fc2 = nn.Linear(layer_2_size, encodingSize)
        self.fc3 = nn.Linear(encodingSize, layer_2_size)
        self.fc4 = nn.Linear(layer_2_size, input_output_size)

        self.lossCriterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.02)
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

def prepareTrainingData(rawTrainingData, encoder: AbstractEncoder):
    for i in rawTrainingData:
        encoder.encode(i[0])
    data_X = [encoder.encode(i[0]) for i in rawTrainingData]
    data_y = [encoder.encode(i[1]) for i in rawTrainingData]

    # TODO Trainer is broken! Pls fix
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)
    X_train = Tensor(X_train).to(device)
    X_test = Tensor(X_test).to(device)
    y_train = Tensor(y_train).to(device)
    y_test = Tensor(y_test).to(device)

    return (X_train, X_test, y_train, y_test)

def trainAutoEncoder(network: EncoderTrainingNetwork, X_train: Tensor, y_train: Tensor, batch_size = 20):
    for batch_start_index in range(0, len(X_train), batch_size):
        x = X_train[batch_start_index:batch_start_index + batch_size]
        y = y_train[batch_start_index:batch_start_index + batch_size]
        network.training_step(x, y)
    return network

def trainAndValidateAutoEncoder(network: EncoderTrainingNetwork, data: Tuple[Tensor, Tensor, Tensor, Tensor], batch_size, cycles)\
        -> Tuple[List[float], List[int], List[float]]:
    (X_train, X_test, y_train, y_test) = data
    validationLoss = []
    validationLossGenerations = []

    validationResults = network(X_test)
    validationLoss.append(float(network.lossCriterion(validationResults, y_test)))
    validationLossGenerations.append(network.generation)

    for i in range(cycles):
        trainAutoEncoder(network, X_train, y_train, batch_size)

        validationResults = network(X_test)
        validationLoss.append(float(network.lossCriterion(validationResults, y_test)))
        validationLossGenerations.append(network.generation)
        print(f"\rTraining cycle {i+1}/{cycles} complete.", end="")

    print()
    return (network.loss_per_step, validationLossGenerations, validationLoss)

def plotResults(results: Tuple[List[float], List[int], List[float]], generation: int, legend, color='blue', valcolor='black'):
    (loss, valLossGenerations, valLoss) = results
    x_loss = list(range(1, generation + 1))
    plt.plot(x_loss, loss, color=color, label=legend)
    plt.plot(valLossGenerations, valLoss, color=valcolor, linestyle='dashed')
    pass


def trainOneHot():
    rawTrainingData = loadTrainingData()
    print("Training data loaded.")
    one_hot = OneHotEncoder()
    one_hot_data = prepareTrainingData(rawTrainingData, one_hot)
    print("Training data encoded.")

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=one_hot.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, one_hot_data, batch_size=10, cycles=1)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 10, 1 cycle", color='green')

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=one_hot.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, one_hot_data, batch_size=150, cycles=15)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 150, 15 cycles", color='red')

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=one_hot.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, one_hot_data, batch_size=350, cycles=35)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 350, 35 cycles", color='blue')

    plt.title("Using a One-Hot Encoder")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0.0009, 0.0017)
    plt.show()

def trainBinary():
    rawTrainingData = loadTrainingData()
    print("Training data loaded.")
    binary = BinaryEncoder()
    binary_data = prepareTrainingData(rawTrainingData, binary)
    print("Training data encoded.")

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=binary.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, binary_data, batch_size=10, cycles=1)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 10, 1 cycle", color='green')

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=binary.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, binary_data, batch_size=150, cycles=15)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 150, 15 cycles", color='red')

    encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=binary.get_encoding_size()).to(device)
    trainingResults = trainAndValidateAutoEncoder(encoderTrainingNetwork, binary_data, batch_size=350, cycles=35)
    plotResults(trainingResults, encoderTrainingNetwork.generation, legend="Batch size 350, 35 cycles", color='blue')

    plt.title("Using a Binary Encoder")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0.1, 0.4)
    plt.show()


if __name__ == '__main__':
    trainOneHot()
    #trainBinary()