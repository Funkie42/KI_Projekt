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
from datasets.edge_vector_dataset import EdgeVectorDataset
from graph import Graph
from node import Node
from part import Part

encoder = auto_encoder.loadPretrainedAutoEncoder()

input_dim = 101
hidden_dim = 100
output_dim = 30

base_dataset = EdgeVectorDataset(part_encoder=encoder)




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

                # Get predictions from the maximum value)
                _, predicted_edge_node_ids = torch.topk(outputs.data, dim=0, k=2)

                # Get the actual node edge ids
                _, edge_node_ids = torch.topk(validate, dim=0, k=2)

                # Total number of labels
                total += validate.size(0)

                # Total correct predictions
                for edge in validate:
                    pass



                correct += (predicted == validate).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))




