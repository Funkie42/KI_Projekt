import pickle
from typing import Set, List

import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

from config.config import root_path, device
from encoder.abstract_encoder import AbstractEncoder
from encoder.one_hot_encoder import OneHotEncoder
from graph import Graph
from models.abstract_prediction_model import MyPredictionModel
from models.ffnn.feedforward_neural_network import FeedforwardNeuralNetwork
from part import Part
from util.graph_utils import toEncodedPartPairs_fromParts, optimize_edge_vector, construct_graph_from_edge_vector


# The
class FFNNGraphPredictionModel(MyPredictionModel):

    def __init__(self, file_path: str, encoder: AbstractEncoder):
        self.encoder = encoder

        input_dim = encoder.get_encoding_size() * 2
        hidden_dim_1 = 402
        hidden_dim_2 = 101
        hidden_dim_3 = 10
        output_dim = 1

        self.network = FeedforwardNeuralNetwork(input_dim, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3).to(device)
        self.network.load_state_dict(torch.load(file_path, map_location=device))
        self.network.eval()


    def predict_graph(self, parts: Set[Part]) -> Graph:
        parts = list(parts)
        encoded_parts_pair_tensor = toEncodedPartPairs_fromParts(parts, self.encoder)

        predicted_edge_vector = self.network(encoded_parts_pair_tensor)
        predicted_edge_vector = optimize_edge_vector(predicted_edge_vector)
        predicted_edge_vector = predicted_edge_vector.to(device)

        return construct_graph_from_edge_vector(parts, predicted_edge_vector)
'''
if __name__ == '__main__':
    # Lade die train data
    with open(f'{root_path}/data/graphs.dat', 'rb') as file:
        graphs: List[Graph] = pickle.load(file)

    # Lade das finale Model
    model_file_path = f"{root_path}/data/trained_ffnn_50_epochs.dat"
    encoder = OneHotEncoder()

    # Splite die Daten und nehme nur Testdaten her. Gleicher Random_state wie im Training
    _, train_graphs = train_test_split(graphs, random_state=1, test_size=0.1)

    #Lade das Modell
    prediction_model: MyPredictionModel = FFNNGraphPredictionModel(model_file_path, encoder)

    # Test der Score
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:100]]
    eval_score = evaluate(prediction_model, instances)
    print(f"Edge accuracy: {round(eval_score, 1)}%")'''