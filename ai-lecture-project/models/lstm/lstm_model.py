import pickle
from typing import Set, List

import torch
from torch import Tensor

from config.config import root_path, device
from encoder.abstract_encoder import AbstractEncoder
from encoder.one_hot_encoder import OneHotEncoder
from models.abstract_prediction_model import MyPredictionModel
from graph import Graph
from models.lstm.lstm_network import LSTMGraphPredictor, calcAdjMatrixAxisSizeFromEdgeVectorSize, adjMatrixToEdgeVector
from part import Part


def optimize_edge_vector(edge_vector: Tensor) -> Tensor:
    """
    Optimizes the edge vector by directly building a non-cyclic but fully connected graph using edges which
    have the highest edge probability.
    """
    part_count = calcAdjMatrixAxisSizeFromEdgeVectorSize(len(edge_vector))

    subgraph_by_part = []
    for i in range(part_count):
        subgraph_by_part.append({i})

    edge_probabilities = []  # List of 3-tuples. The first two tuple elements are node ids, the third is the edge probability
    index = 0
    for i in range(part_count - 1):
        for j in range(i + 1, part_count):
            edge_probabilities.append((i, j, edge_vector[index]))
            index += 1

    edge_probabilities.sort(key=lambda x: x[2], reverse=True)

    adj_matrix = torch.zeros((part_count, part_count))

    for (n1, n2, _) in edge_probabilities:
        n1_subgraph = subgraph_by_part[n1]
        n2_subgraph = subgraph_by_part[n2]
        if n1_subgraph != n2_subgraph:
            adj_matrix[n1, n2] = 1
            adj_matrix[n2, n1] = 1
            combined_subgraph = n1_subgraph.union(n2_subgraph)
            for node_index in combined_subgraph:
                subgraph_by_part[node_index] = combined_subgraph

    return adjMatrixToEdgeVector(adj_matrix)

class LSTMGraphPredictionModel(MyPredictionModel):

    def __init__(self, encoder: AbstractEncoder = OneHotEncoder(), file_path = None):
        self.encoder = encoder
        self.network = LSTMGraphPredictor(encoder.get_encoding_size()).to(device)
        if file_path != None:
            self.network.load_state_dict(torch.load(file_path, map_location=device))
        self.network.eval()

    def construct_graph_from_edge_vector(self, parts: List[Part], edge_vector: Tensor) -> Graph:
        graph = Graph()
        index = 0
        for p1 in range(len(parts) - 1):
            for p2 in range(p1 + 1, len(parts)):
                edge_probability = edge_vector[index]
                if edge_probability >= 0.5:
                    graph.add_undirected_edge(parts[p1], parts[p2])
                index += 1
        return graph


    def predict_graph(self, parts: Set[Part]) -> Graph:
        parts = list(parts)
        encoded_parts_list = [self.encoder.encode(p) for p in parts]
        encoded_parts_tensor = torch.stack(encoded_parts_list).to(device)

        predicted_edge_vector = self.network(encoded_parts_tensor)
        predicted_edge_vector = optimize_edge_vector(predicted_edge_vector)
        predicted_edge_vector = predicted_edge_vector.to("cpu")

        return self.construct_graph_from_edge_vector(parts, predicted_edge_vector)

if __name__ == '__main__':
    # Load train data
    with open(f'{root_path}/data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    # Load the final model

    model_file_path = None # f"{root_path}/data/trained_lstm.dat"
    encoder = OneHotEncoder()
    prediction_model: MyPredictionModel = LSTMGraphPredictionModel(encoder)

    # For illustration, compute eval score on train data
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:100]]
    #eval_score = evaluate(prediction_model, instances)
    #print(f"Edge accuracy: {round(eval_score, 1)}%")