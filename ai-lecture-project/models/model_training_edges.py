from typing import List, Dict
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim, Tensor
import pickle
from sklearn.model_selection import train_test_split


import ffnn.feedforward_neural_network_model as ffnn
import encoder.auto_encoder_trainer as encoder_trainer
import encoder.auto_encoder_decoder as auto_encoder
from config.config import device
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from node import Node
from part import Part

encoder = auto_encoder.loadPretrainedAutoEncoder()
encoder = OneHotEncoder()

input_dim = encoder.get_encoding_size() * 2
hidden_dim_1 = 402
hidden_dim_2 = 101
hidden_dim_3 = 10

output_dim = 1


def toEdgeVector(graph):
    nodelist: List[Node] = list(graph.get_nodes())
    edges: Dict[Node, List[Node]] = graph.get_edges()
    edgeVector = []
    for index, node in enumerate(nodelist):
        node_edges = edges[node]
        for node2 in nodelist[index + 1:]:
            if node2 in node_edges:
                edgeVector.append(1.0)
            else:
                edgeVector.append(0.0)
    return torch.Tensor(edgeVector)

def toEncodedPartPairs(graph):
    nodelist: List[Node] = list(graph.get_nodes())
    encoded_nodelist: List[Node] = []
    for node in nodelist:
        encoded_nodelist.append(encoder.encode(node.get_part()))
    encoded_part_pairs = []
    for index, part_tensor in enumerate(encoded_nodelist):
        for part2_tensor in encoded_nodelist[index + 1:]:
            part_pair_tensor = torch.cat((part_tensor, part2_tensor))
            encoded_part_pairs.append(part_pair_tensor)
    return torch.stack(encoded_part_pairs)



def constructModelTrainingData() -> (Tensor, Tensor):
    """
    Constructs the data used to train the model.
    """
    with open('../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    input = []
    result = []
    for g in train_graphs:
        result_node_pairs = toEdgeVector(g)
        input_node_pairs = toEncodedPartPairs(g)
        input.append(input_node_pairs)
        result.append(result_node_pairs)

    return input, result

def prepareData():
    data_X, data_y = constructModelTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepareData()

print("Data prepared and Nodes encoded.")

batch_size = 100
n_iter = 3000
n_epochs = int(n_iter / (len(X_train) / batch_size))

model = ffnn.FeedforwardNeuralNetworkModel(input_dim,output_dim,hidden_dim_1,hidden_dim_2, hidden_dim_3)

criterion = nn.BCELoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iterator = 0

print("Start training")

for epoch in range(n_epochs):
    for (input, validate) in zip(*(X_train, y_train)):
        optimizer.zero_grad()
        outputs = model(input).to(device)
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
                outputs = model(input)

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




