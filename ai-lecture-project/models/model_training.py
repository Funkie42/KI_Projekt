from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim, Tensor
import pickle
from sklearn.model_selection import train_test_split


import ffnn.feedforward_neural_network_model as ffnn
import encoder.auto_encoder_trainer as encoder_trainer
import encoder.auto_encoder_decoder as auto_encoder
from graph import Graph
from node import Node
from part import Part

encoder = auto_encoder.loadPretrainedAutoEncoder()

input_dim = 101
hidden_dim = 100
output_dim = 31


def constructModelTrainingData() -> (Tensor, Tensor):
    """
    Constructs the data used to train the model.
    """
    with open('../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    encoded_nodes_tensor = []
    result = []
    for g in train_graphs:
        result_nodes_list = []


        input_nodes = []


        for node in g.get_nodes():
            id_tensor = torch.tensor([node.get_id()])
            part_tensor = encoder.encode(node.get_part())
            node_tensor = torch.cat((id_tensor, part_tensor))
            input_nodes.append(node_tensor)

        for (node, connectedNodes) in g.get_edges().items():
            id_float = float(node.get_id()) / output_dim
            result_node_tensor = torch.cat((torch.tensor([id_float]), torch.zeros(output_dim - 1)))

            for connectedNode in connectedNodes:
                result_node_tensor[connectedNode.get_id() + 1] = 1
                #result_edges.append((node.get_part(), connectedNode.get_part()))

            result_nodes_list.append(result_node_tensor)

        input_nodes_tensor = torch.stack(input_nodes)
        #scaled_input_nodes_tensor = torch.zeros(input_dim, output_dim)
        #original_size = input_nodes_tensor.size()

        #scaled_input_nodes_tensor[:original_size[0], :original_size[1]] = input_nodes_tensor
        encoded_nodes_tensor.append(input_nodes_tensor)

        result_nodes_tensor = torch.stack(result_nodes_list)
        result.append(result_nodes_tensor)



    return encoded_nodes_tensor, result

def prepareData():
    data_X, data_y = constructModelTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepareData()

print("Data prepared and Nodes encoded.")

batch_size = 100
n_iter = 3000
n_epochs = int(n_iter / (len(X_train) / batch_size))



model = ffnn.FeedforwardNeuralNetworkModel(input_dim,output_dim,hidden_dim)


criterion = nn.CrossEntropyLoss() #nn.BCELoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0

print("Start training")

for epoch in range(n_epochs):
    for (input, validate) in zip(*(X_train, y_train)):
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, validate)
        loss.backward()
        optimizer.step()
        iter += 1

        if iter % 100 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for input, validate in zip(*(X_test, y_test)):
                # Load images with gradient accumulation capabilities

                # Forward pass only to get logits/output
                outputs = model(input)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, dim=0)

                # Total number of labels
                total += validate.size(0)

                # Total correct predictions
                correct += (predicted == validate).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))




