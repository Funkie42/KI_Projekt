from typing import List, Dict

import torch

from config.config import device
from datasets.graph_dataset import GraphDataset
from encoder.abstract_encoder import AbstractEncoder
from node import Node


class EdgeVectorDataset(GraphDataset):
    """
    This dataset returns all graphs as lists of parts, possibly pre-encoded with a given encoder.
    The labels for each graph are given as edge vector.
    """

    def __init__(self, part_encoder: AbstractEncoder = None):
        super().__init__()
        self.dataset = []
        for index in range(len(self)):
            (parts, graph) = super().__getitem__(index)
            if part_encoder:
                parts = torch.stack([part_encoder.encode(p) for p in parts]).to(device)

            self.dataset.append((parts, self.toEdgeVector(graph).to(device)))

    def toEdgeVector(self, graph):
        nodelist: List[Node] = list(graph.get_nodes())
        edges: Dict[Node, List[Node]] = graph.get_edges()
        edgeVector = []
        for index, node in enumerate(nodelist):
            node_edges = edges[node]
            for node2 in nodelist[index+1:]:
                if node2 in node_edges:
                    edgeVector.append(1.0)
                else:
                    edgeVector.append(0.0)
        return torch.Tensor(edgeVector)

    def __getitem__(self, idx):
        return self.dataset[idx]

