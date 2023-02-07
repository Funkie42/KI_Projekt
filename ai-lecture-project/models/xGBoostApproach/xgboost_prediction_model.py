import pickle
from itertools import permutations
from typing import List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from evaluation import MyPredictionModel
from graph import Graph
from models.xGBoostApproach.data_converter import DataConverter
from part import Part
import xgboost as xgb
from sklearn.metrics import accuracy_score

from util.graph_utils import optimize_edge_vector


class XGBoostModel(MyPredictionModel):

    def __init__(self, dc: DataConverter):
        self.dc = dc
        self.x_train, self.y_train, self.x_test, self.y_test = dc.get_data()
        self.xg_classifier = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=100)

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
        rows = []
        for i, p1 in enumerate(parts):
            for p2 in parts[i+1:]:
                rows.append(self.dc.get_instance(p1, p2, parts))

        preds = self.xg_classifier.predict(pd.DataFrame(rows))
        edge_tensor = optimize_edge_vector(torch.Tensor(preds))

        return self.construct_graph_from_edge_vector(parts, edge_tensor)

    def train(self):
        self.xg_classifier.fit(self.x_train, self.y_train)

    def evaluate(self):
        y_pred = self.xg_classifier.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))


def evaluate(model: MyPredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> float:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0
    n_graphs = 1
    for input_parts, target_graph in data_set:
        predicted_graph = model.predict_graph(input_parts)

        edges_counter += len(input_parts) * len(input_parts)
        sum_correct_edges += edge_accuracy(predicted_graph, target_graph)
        print(n_graphs)
        n_graphs += 1
        # FYI: maybe some more evaluation metrics will be used in final evaluation

    return sum_correct_edges / edges_counter * 100


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'
    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(predicted_graph.get_parts())
    # larger graphs take way too long because of the many permutations
    # just return "50% accuracy"
    if len(perms) > 10000:
        return len(predicted_graph.get_parts()) * len(predicted_graph.get_parts()) // 2
    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for i, perm in enumerate(perms):
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)
    return best_score


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all(
        [len(perm) == len(parts) for perm in full_perms]), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Example code for evaluation

if __name__ == '__main__':
    # Load train data
    with open('../../data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    all_parts = [p for parts in map(lambda g: g.get_parts(), train_graphs) for p in parts]
    model = XGBoostModel(DataConverter(train_graphs, all_parts, save_instances_version=20, load_instances_version=None))
    model.train()
    model.evaluate()
    # For illustration, compute eval score on train data
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:1000]]
    eval_score = evaluate(model, instances)
    print(eval_score)
