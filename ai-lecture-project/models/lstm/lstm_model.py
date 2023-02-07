import pickle
from typing import Set, List

import torch
from torch import Tensor

from config.config import root_path, device
from encoder.abstract_encoder import AbstractEncoder
from encoder.one_hot_encoder import OneHotEncoder
from evaluation import MyPredictionModel, evaluate
from graph import Graph
from models.lstm.lstm_network import LSTMGraphPredictor, calcAdjMatrixAxisSizeFromEdgeVectorSize, adjMatrixToEdgeVector
from part import Part
from util.graph_utils import *


class LSTMGraphPredictionModel(MyPredictionModel):

    def __init__(self, file_path: str, encoder: AbstractEncoder):
        self.encoder = encoder
        self.network = LSTMGraphPredictor(encoder.get_encoding_size()).to(device)
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

    model_file_path = f"{root_path}/data/trained_lstm_50_epochs.dat"
    encoder = OneHotEncoder()
    prediction_model: MyPredictionModel = LSTMGraphPredictionModel(model_file_path, encoder)

    # For illustration, compute eval score on train data
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:100]]
    eval_score = evaluate(prediction_model, instances)
    print(f"Edge accuracy: {round(eval_score, 1)}%")