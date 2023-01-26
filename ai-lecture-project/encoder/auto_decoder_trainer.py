import pickle
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
import torch.nn.functional as F

from config.config import device, auto_encoder_encoding_size, auto_encoder_training_intermediate_layer_size
from encoder.abstract_encoder import AbstractEncoder
from encoder.auto_encoder import AutoEncoder
from encoder.auto_encoder_trainer import TrainingNetwork, Trainer
from graph import Graph


class DecoderTrainingNetwork(TrainingNetwork):

    def __init__(self, input_output_size: int):
        super(DecoderTrainingNetwork, self).__init__()

        self.fc1 = nn.Linear(auto_encoder_encoding_size, auto_encoder_training_intermediate_layer_size)
        self.fc2 = nn.Linear(auto_encoder_training_intermediate_layer_size, input_output_size)

        self.lateinit()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def constructTrainingData(rawData: List[Graph], preEncoder: AbstractEncoder, encoder: AutoEncoder) -> List[Tuple[Tensor, Tensor]]:
    """
    Constructs the data used to train the encoder. Returns a list of tuples, where each tuple consists of two tensors.
    The first item of each tuple is a heavily auto-encoded part and the second one is the only pre-encoded part.
    """
    result = []

    graph = 1
    for g in rawData:
        print(f"\rLoading and encoding parts from graph {graph}/{len(rawData)} to train decoder...", end="")
        graph += 1
        for node in g.get_nodes():
            encodedPart = preEncoder.encode(node.get_part()).to(device)
            result.append((encoder.encodeTensor(encodedPart), encodedPart))
    print()
    return result

def loadTrainingData(preEncoder: AbstractEncoder, encoder: AutoEncoder):
    with open('../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    return constructTrainingData(train_graphs, preEncoder, encoder)

class DecoderTrainer(Trainer):

    def __init__(self, preEncoder: AbstractEncoder, encoder: AutoEncoder):
        super().__init__()
        self.preEncoder = preEncoder
        self.encoder = encoder
        self.rawTrainingData = loadTrainingData(preEncoder, encoder)

        data_X = [t[0] for t in self.rawTrainingData]
        data_y = [t[1] for t in self.rawTrainingData]
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

        self.X_train = torch.stack(X_train).to(device)
        self.X_test = torch.stack(X_test).to(device)
        self.y_train = torch.stack(y_train).to(device)
        self.y_test = torch.stack(y_test).to(device)

        print("Decoder training data loaded.")

    def trainAndPlotResults(self, batch_size, cycles, legend=None, color='blue') -> DecoderTrainingNetwork:
        decoderTrainingNetwork = DecoderTrainingNetwork(input_output_size=self.preEncoder.get_encoding_size()).to(device)
        trainingResults = self.trainAndValidate(decoderTrainingNetwork, batch_size=batch_size, cycles=cycles)
        self.plotResults(trainingResults, decoderTrainingNetwork.generation, legend=legend if legend != None else '', color=color)

        return decoderTrainingNetwork
