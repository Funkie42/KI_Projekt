from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import pickle
from sklearn.model_selection import train_test_split


import models.ffnn.feedforward_neural_network as ffnn
import encoder.auto_encoder_decoder as auto_encoder
from config.config import device, root_path
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from util.graph_utils import toEdgeVector, toEncodedPartPairs


# Hier wird das Model für FFNN trainiert.
# Wir verwenden OneHot Encoding für die Encodierung
encoder = OneHotEncoder()

# Definieren der Layer Dimensionen
# Input Dim ist 2*Encoding Größe, da wir immer nach allen edges von Parts trainieren
input_dim = encoder.get_encoding_size() * 2
hidden_dim_1 = 402
hidden_dim_2 = 101
hidden_dim_3 = 10
output_dim = 1

# Epochenzahl für das Training
# Trainingseffekt ist für Epoche 3-50 nur minimal (Verbesserung von 95%.. acc auf 96%.. acc)
n_epochs = 50

def constructModelTrainingData() -> (List[Tensor], List[Tensor]):
    """
    Constructs the data used to train the model.
    Returns 2 lists of tensors, Input and Output tensor lists
    Input is a list of tensors is of shape 'edges' (100 + 100 for both encodings)
    Output is a list of tensors is of shape 'edges-isEdgeBool', which is used for loss fct and training
    """
    # Graphen laden
    with open('../../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
    input = []
    result = []
    # Für jeden Graphen:
    for g in train_graphs:
        #output paare laden
        result_node_pairs = toEdgeVector(g)
        #input paare laden
        input_node_pairs = toEncodedPartPairs(g, encoder)

        input.append(input_node_pairs)
        result.append(result_node_pairs)

    return input, result

def prepareData():
    '''
    Einteilung der Daten in Training und Testing
    Output und Input weiterhin getrennt
    '''
    data_X, data_y = constructModelTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1, test_size=0.2)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    # Daten vorbereiten
    X_train, X_test, y_train, y_test = prepareData()

    print("Data prepared and Nodes encoded.")


    # Netzwerk initieren
    network = ffnn.FeedforwardNeuralNetwork(input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3)

    # Wir nutzen die Binary Cross Entropy als Loss funktion
    criterion = nn.BCELoss()

    learning_rate = 0.1

    #Stochastic Gradient Descent für den Optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    iterator = 0

    #Nur zum prüfen wie es besser wird langfristig
    print("Start training")
    accuracy_improvement = []
    old_acc = 0

    # Training über Epochen
    for epoch in range(n_epochs):
        # Zip input und output für Iterator
        for (input, validate) in zip(*(X_train, y_train)):
            # Setze gradienten tensor to zero
            optimizer.zero_grad()
            # Durchlaufen durch das Netzwerk
            outputs = network(input).to(device)
            # Loss Berechnung, unsqueeze den output um gleiches format zu haben der tensoren
            loss = criterion(outputs, torch.unsqueeze(validate, dim=1))
            # Backward Propagation
            loss.backward()
            # Parameter anpassen des Optimizers
            optimizer.step()
            iterator += 1

            # Testen des Fortschritt
            if iterator % 100 == 0:
                # Langfristige Verbesserung testen
                if (iterator % 2000 == 1900):
                    acc = sum(accuracy_improvement) / len(accuracy_improvement)

                    print(f"Old accuracy: {old_acc}, new accuracy: {acc}")
                    accuracy_improvement = []
                    old_acc = acc
                # Calculate Accuracy
                correct = 0
                total = 0

                correct_existing_edges = 0
                existing_edges = 0
                # Iterate through test dataset
                for input, validate in zip(*(X_test, y_test)):

                    # Nur Forward Netzwerk ohne Backpropagation da test ohne training
                    outputs = network(input)

                    # Prediction Prüfung
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
                        total += 1

                accuracy = 100 * correct / total
                real_edge_accuracy = 100 * correct_existing_edges / existing_edges

                accuracy_improvement.append(accuracy)

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}. Real Edge Accuracy: {}.'.format(iterator, loss.item(), accuracy, real_edge_accuracy))

    # Save the network
    torch.save(network.state_dict(), f'{root_path}/data/trained_ffnn_{n_epochs}_epochs.dat')



