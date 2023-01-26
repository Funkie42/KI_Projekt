import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim, Tensor

from config.config import device, auto_encoder_training_intermediate_layer_size, auto_encoder_encoding_size
from encoder.abstract_encoder import AbstractEncoder
from encoder.binary_encoder import BinaryEncoder
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from part import Part


# Known Id Range:
# Part id: 0 - 2270 (both inclusive)
# Family id: 0 - 100 (both inclusive)

class TrainingNetwork(nn.Module):

    def __init__(self):
        super(TrainingNetwork, self).__init__()

    def lateinit(self):
        self.lossCriterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.02)
        self.loss_per_step = []
        self.generation = 0

    def training_step(self, data_x, data_y):
        self.optimizer.zero_grad()
        outputs = self(data_x)
        loss = self.lossCriterion(outputs, data_y)
        self.loss_per_step.append(float(loss))
        loss.backward()
        self.optimizer.step()
        self.generation += 1
        return loss


class EncoderTrainingNetwork(TrainingNetwork):

    def __init__(self, input_output_size: int):
        super(EncoderTrainingNetwork, self).__init__()

        self.fc1 = nn.Linear(input_output_size, auto_encoder_training_intermediate_layer_size)
        self.fc2 = nn.Linear(auto_encoder_training_intermediate_layer_size, auto_encoder_encoding_size)
        self.fc3 = nn.Linear(auto_encoder_encoding_size, auto_encoder_training_intermediate_layer_size)
        self.fc4 = nn.Linear(auto_encoder_training_intermediate_layer_size, input_output_size)

        self.lateinit()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class Trainer:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None

    def trainCycle(self, network: TrainingNetwork, X_train: Tensor, y_train: Tensor, batch_size = 20):
        for batch_start_index in range(0, len(X_train), batch_size):
            x = X_train[batch_start_index:batch_start_index + batch_size]
            y = y_train[batch_start_index:batch_start_index + batch_size]
            network.training_step(x, y)
        return network

    def trainAndValidate(self, network: TrainingNetwork, batch_size, cycles)\
            -> Tuple[List[float], List[int], List[float]]:
        validationLoss = []
        validationLossGenerations = []

        validationResults = network(self.X_test)
        validationLoss.append(float(network.lossCriterion(validationResults, self.y_test)))
        validationLossGenerations.append(network.generation)

        for i in range(cycles):
            self.trainCycle(network, self.X_train, self.y_train, batch_size)

            validationResults = network(self.X_test)
            validationLoss.append(float(network.lossCriterion(validationResults, self.y_test)))
            validationLossGenerations.append(network.generation)
            print(f"\rTraining cycle {i+1}/{cycles} complete.", end="")

        print()
        return (network.loss_per_step, validationLossGenerations, validationLoss)

    def plotResults(self, results: Tuple[List[float], List[int], List[float]], generation: int, legend="", color='blue', valcolor='black'):
        (loss, valLossGenerations, valLoss) = results
        x_loss = list(range(1, generation + 1))
        plt.plot(x_loss, loss, color=color, label=legend)
        plt.plot(valLossGenerations, valLoss, color=valcolor, linestyle='dashed')
        pass

class EncoderTrainer(Trainer):

    def __init__(self, rawTrainingData):
        super().__init__()
        self.rawTrainingData = rawTrainingData

    def prepareTrainingData(self, encoder: AbstractEncoder):
        self.encoder = encoder
        data_X = [encoder.encode(i[0]) for i in self.rawTrainingData]
        data_y = [encoder.encode(i[1]) for i in self.rawTrainingData]

        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

        self.X_train = torch.stack(X_train).to(device)
        self.X_test = torch.stack(X_test).to(device)
        self.y_train = torch.stack(y_train).to(device)
        self.y_test = torch.stack(y_test).to(device)

        return (self.X_train, self.X_test, self.y_train, self.y_test)

    def trainAndPlotResults(self, batch_size, cycles, legend='', color='blue') -> EncoderTrainingNetwork:
        encoderTrainingNetwork = EncoderTrainingNetwork(input_output_size=self.encoder.get_encoding_size()).to(device)
        trainingResults = self.trainAndValidate(encoderTrainingNetwork, batch_size=batch_size, cycles=cycles)
        self.plotResults(trainingResults, encoderTrainingNetwork.generation, legend=legend, color=color)
        return encoderTrainingNetwork

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

def trainOneHot(trainer: EncoderTrainer):
    trainer.prepareTrainingData(OneHotEncoder())
    print("Training data encoded.")

    trainer.trainAndPlotResults(10, 5, "Batch size 10, 5 cycles", color='green')
    trainer.trainAndPlotResults(150, 75, "Batch size 150, 75 cycles", color='red')
    trainer.trainAndPlotResults(350, 175, "Batch size 350, 175 cycles", color='blue')

    plt.title("Using a One-Hot Encoder")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0.0006, 0.0017)
    plt.show()


def trainBinary(trainer: EncoderTrainer):
    trainer.prepareTrainingData(BinaryEncoder())
    print("Training data encoded.")

    trainer.trainAndPlotResults(10, 1, "Batch size 10, 1 cycle", color='green')
    trainer.trainAndPlotResults(150, 15, "Batch size 150, 15 cycles", color='red')
    trainer.trainAndPlotResults(350, 35, "Batch size 350, 35 cycles", color='blue')

    plt.title("Using a Binary Encoder")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0.1, 0.4)
    plt.show()

if __name__ == '__main__':
    rawTrainingData = loadTrainingData()
    print("Encoder training data loaded.")

    trainer = EncoderTrainer(rawTrainingData)

    trainOneHot(trainer)
    #trainBinary(trainer)

    #trainAndSaveAutoEncoder(rawTrainingData)