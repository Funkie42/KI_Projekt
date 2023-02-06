import math
from typing import List, Dict, Set

import torch
from torch import Tensor

from node import Node
from part import Part


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

def toEncodedPartPairs(graph, encoder):
    nodelist: List[Node] = list(graph.get_nodes())
    encoded_nodelist = []
    for node in nodelist:
        encoded_nodelist.append(encoder.encode(node.get_part()))
    encoded_part_pairs = []
    for index, part_tensor in enumerate(encoded_nodelist):
        for part2_tensor in encoded_nodelist[index + 1:]:
            part_pair_tensor = torch.cat((part_tensor, part2_tensor))
            encoded_part_pairs.append(part_pair_tensor)
    return torch.stack(encoded_part_pairs)

def toEncodedPartPairs_fromParts(parts: Set[Part], encoder):
    partlist: List[Part] = list(parts)
    encoded_partlist = []
    for part in partlist:
        encoded_partlist.append(encoder.encode(part))
    encoded_part_pairs = []
    for index, part_tensor in enumerate(encoded_partlist):
        for part2_tensor in encoded_partlist[index + 1:]:
            part_pair_tensor = torch.cat((part_tensor, part2_tensor))
            encoded_part_pairs.append(part_pair_tensor)
    return torch.stack(encoded_part_pairs)

def calcEdgeVectorSizeFromAdjMatrixAxisSize(adjMatrixAxisSize: int) -> int:
    return (adjMatrixAxisSize ** 2 - adjMatrixAxisSize) // 2


def calcAdjMatrixAxisSizeFromEdgeVectorSize(edgeVectorSize: int) -> int:
    return math.isqrt(2 * edgeVectorSize) + 1


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

def adjMatrixToEdgeVector(adj_matrix: Tensor) -> Tensor:
    """
    Turns the given adjacency matrix into an edge vector. You may also pass a batch of adjacency matrices
    to get a batch of edge vectors.
    """
    unsqueezed = False
    if len(adj_matrix.shape) < 3:
        adj_matrix = adj_matrix.unsqueeze(0)
        unsqueezed = True

    axis_size = adj_matrix.shape[2]

    if axis_size != adj_matrix.shape[1]:
        raise RuntimeError(
            "The given adjacency matrix's dimensions are not of the same size! How can this be an adjacency matrix??")

    edge_vector = []
    for i1 in range(axis_size - 1):
        for i2 in range(i1 + 1, axis_size):
            edge_vector.append(adj_matrix[:, i1, i2])

    edge_vector = torch.stack(edge_vector).T

    if unsqueezed:
        edge_vector = edge_vector.squeeze(0)
    return edge_vector


