import pickle
from itertools import permutations
from typing import List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config.config import root_path
from graph import Graph
from models.abstract_prediction_model import MyPredictionModel
from models.xGBoostApproach.data_converter import DataConverter
from part import Part
import xgboost as xgb
from sklearn.metrics import accuracy_score

from util.graph_utils import optimize_edge_vector, construct_graph_from_edge_vector

# from evaluation import evaluate


class XGBoostModel(MyPredictionModel):
    """
    This class stores a DataConverter and an XGBoostClassifier object to make predictions
    on Part pairs in a Graph.
    """
    def __init__(self, dc: DataConverter):
        self.dc = dc
        self.x_train, self.y_train, self.x_test, self.y_test = dc.get_data()
        # Use an XGBoostClassifier to predict if two parts are connected by an edge
        # given extra information specified in DataConverter.get_instance()
        self.xg_classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=2000)

    def predict_graph(self, parts: Set[Part]) -> Graph:
        parts = list(parts)
        rows = []
        for i, p1 in enumerate(parts):
            for p2 in parts[i+1:]:
                # Feed all pairs of parts into the classifier
                rows.append(self.dc.get_instance(p1, p2, parts))

        preds = self.xg_classifier.predict(pd.DataFrame(rows))
        # Make sure the graph is connected
        edge_tensor = optimize_edge_vector(torch.Tensor(preds))
        # Return a Graph object generated from the edge-vector
        return construct_graph_from_edge_vector(parts=parts, edge_vector=edge_tensor)

    def train(self):
        self.xg_classifier.fit(self.x_train, self.y_train)

    def evaluate(self):
        # If a dataset was passed to the DataConverter, the model can evaluate itself
        # (Just how many instances were classified correctly, not if the graphs have the
        # wanted properties)
        if self.x_test is None:
            print("No dataset to test on")
            return
        y_pred = self.xg_classifier.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def save(self, name):
        model_file_path = f"{root_path}/data/trained_xgboost_{name}.dat"
        with open(model_file_path, "wb") as f:
            pickle.dump(self.xg_classifier, f)
        print("Saved model ", model_file_path)

    def load(self, name):
        model_file_path = f"{root_path}/data/trained_xgboost_{name}.dat"
        with open(model_file_path, "rb") as f:
            self.xg_classifier = pickle.load(f)
        print("Loaded model ", model_file_path)


if __name__ == '__main__':
    # Load train data
    with open(f'{root_path}/data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    dc = DataConverter(train_graphs, save_instances_version=None, load_instances_version=40)
    model = XGBoostModel(dc)

    model.train()
    model.save('model_2000_estimators')

    # model.load('model1')

    model.evaluate()

    # instances = [(graph.get_parts(), graph) for graph in train_graphs[:1000]]
    # eval_score = evaluate(model, instances)
    # print(eval_score)
