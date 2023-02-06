from typing import List, Dict
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim, Tensor
import pickle
from sklearn.model_selection import train_test_split


import models.ffnn.feedforward_neural_network as ffnn
import encoder.auto_encoder_trainer as encoder_trainer
import encoder.auto_encoder_decoder as auto_encoder
from config.config import device, root_path
from encoder.binary_encoder import BinaryEncoder
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from node import Node
from part import Part
from util.graph_utils import toEdgeVector, toEncodedPartPairs

encoder = auto_encoder.loadPretrainedAutoEncoder()
encoder = OneHotEncoder()

input_dim = encoder.get_encoding_size() * 2
hidden_dim_1 = 402
hidden_dim_2 = 101
hidden_dim_3 = 10
output_dim = 1

n_epochs = 4






def constructModelTrainingData() -> (Tensor, Tensor):
    """
    Constructs the data used to train the model.
    """
    with open('../../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    input = []
    result = []
    for g in train_graphs:
        result_node_pairs = toEdgeVector(g)
        input_node_pairs = toEncodedPartPairs(g, encoder)
        input.append(input_node_pairs)
        result.append(result_node_pairs)

    return input, result

def prepareData():
    data_X, data_y = constructModelTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = prepareData()

    print("Data prepared and Nodes encoded.")



    network = ffnn.FeedforwardNeuralNetwork(input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3)

    criterion = nn.BCELoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    iterator = 0

    print("Start training")

    for epoch in range(n_epochs):
        for (input, validate) in zip(*(X_train, y_train)):
            optimizer.zero_grad()
            outputs = network(input).to(device)
            loss = criterion(outputs, torch.unsqueeze(validate, dim=1))
            loss.backward()
            optimizer.step()
            iterator += 1

            if iterator % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0

                correct_existing_edges = 0
                existing_edges = 0
                # Iterate through test dataset
                for input, validate in zip(*(X_test, y_test)):

                    # Forward pass only to get output
                    outputs = network(input)

                    for prediction, reality in zip(*(outputs, validate)):
                        predicted_bool = 1 if prediction > 0.4 else 0
                        if reality == predicted_bool:
                            correct += 1
                            if reality == 1:
                                correct_existing_edges += 1
                        else:
                            total = total
                        if reality == 1:
                            existing_edges += 1

                            #print("Reality:",reality,"Prediction:",prediction)
                        total += 1

                accuracy = 100 * correct / total
                real_edge_accuracy = 100 * correct_existing_edges / existing_edges

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}. Real Edge Accuracy: {}.'.format(iterator, loss.item(), accuracy, real_edge_accuracy))
    torch.save(network.state_dict(), f'{root_path}/data/trained_ffnn_{n_epochs}_epochs.dat')



