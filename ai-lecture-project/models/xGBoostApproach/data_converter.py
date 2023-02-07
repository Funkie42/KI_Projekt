from random import random
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import csv

from graph import Graph
from part import Part
from sklearn.model_selection import train_test_split


def create_and_save_parts_ordering(all_parts_list: List[Part]) -> List[str]:
    all_parts_list = sorted(list(dict.fromkeys(map(lambda x: repr(x), all_parts_list))))
    return all_parts_list


class DataConverter:

    def __init__(self, ds: List[Graph], all_parts_list=None, save_instances_version=None, load_instances_version=None):
        if all_parts_list is not None:
            self.parts_ordering_map = create_and_save_parts_ordering(all_parts_list)
        else:
            self.parts_ordering_map = self.load_parts_ordering()
        print("Ordering Map done!")

        if load_instances_version is None:
            self.dataframe, self.labels = self.convert_graph_list_to_dataframe(ds)
        else:
            self.dataframe, self.labels = self.load_dataframe_and_labels(load_instances_version)

        if save_instances_version is not None:
            print("Saving dataset with version ", save_instances_version)
            self.dataframe.to_csv(f"./data/instances{save_instances_version}.csv")
            self.labels.to_csv(f"./data/labels{save_instances_version}.csv")

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.dataframe, self.labels, test_size=0.2, random_state=123)

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def load_parts_ordering(self) -> List[str]:
        ...

    def load_dataframe_and_labels(self, version) -> Tuple[pd.DataFrame]:
        print("Loading dataset with version", version)
        return pd.read_csv(f"./data/instances{version}.csv"), pd.read_csv(f"./data/labels{version}.csv")

    def convert_graph_list_to_dataframe(self, ds: List[Graph]) -> pd.DataFrame:
        """
        Converts the training data - a list of graph objects - to a pandas dataframe of the following format:

            node_a_embedding ; node_b_embedding ; label_is_edge ; n_parts_in_graph ; n_parts_emb1 ; n_parts_emb2 ; ...

        where two nodes a and b should be connected iff label_is_edge is true.
        @param ds: the dataset
        """
        instances: List[List] = []
        labels: List[List[bool]] = []
        index = 0
        graphs = 1
        for g in ds:
            print(graphs)
            graphs += 1
            edges = g.get_edges()
            for i, n1 in enumerate(g.get_nodes()):
                for j, n2 in enumerate(g.get_nodes()):
                    if i == j:
                        continue
                    has_edge = n2 in edges[n1]
                    if has_edge or random() < 0.5:
                        row = self.get_instance(n1.get_part(), n2.get_part(), g.get_parts())
                        instances.append(row)
                        labels.append([1 if has_edge else 0])
                        index += 1

        return pd.DataFrame(instances), pd.DataFrame(labels)

    def get_instance(self, p1: Part, p2: Part, parts: Set[Part]) -> [Tuple[int]]:
        family_occs = [0] * 105  # 105 different familiy_id values
        for p in parts:
            index = int(self.part_to_index(p))
            family_occs[index] += 1

        front = [self.part_to_index(p1), self.part_to_index(p2), len(parts)]
        front.extend(family_occs)
        return front

    # def get_instance(self, p1: Part, p2: Part, parts: Set[Part]) -> [Tuple[int]]:
    #     return [self.part_to_index(p1), self.part_to_index(p2), len(parts)]

    def part_to_index(self, p: Part) -> int:
        # return self.parts_ordering_map.index(repr(p))
        return int(p.get_family_id())
