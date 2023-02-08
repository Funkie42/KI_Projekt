from random import random
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import csv

from graph import Graph
from part import Part
from sklearn.model_selection import train_test_split


class DataConverter:
    """
    The DataConverter maps Graph objects (or rather Part-Sets) to a number of rows for the training
    dataset of the XGBoost Classifier. Each Part is just represented by its family, which is already
    enough to describe sufficient similarity between parts. The x values are created by looking at each
    part pair of a graph/Part-Set and saving their family ids, the number of nodes in the graph/Part-Set, and the number
    of occurrences of all other parts (=>part families) in the graph/Part-Set. The labels are then just
    a List of true or false values corresponding to the part pairs describing if the pair should be
    connected by an edge or not.
    """
    def __init__(self, ds: List[Graph], save_instances_version=None, load_instances_version=None):
        # To save time all rows can be saved after they have been calculated and loaded here so they
        # don't have to be calculated again
        if load_instances_version is None:
            self.dataframe, self.labels = self.convert_graph_list_to_dataframe(ds)
        else:
            self.dataframe, self.labels = self.load_dataframe_and_labels(load_instances_version)

        # Maybe save the calculated (or also loaded, to be exact) rows to a file
        if save_instances_version is not None:
            print("Saving dataset with version ", save_instances_version)
            self.dataframe.to_csv(f"./data/instances{save_instances_version}.csv",index_label=False)
            self.labels.to_csv(f"./data/labels{save_instances_version}.csv",index_label=False)

        # Split the dataset if enough entries exist
        if ds is not None and len(ds) >= 5:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self.dataframe, self.labels, test_size=0.2, random_state=123)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def load_dataframe_and_labels(self, version) -> Tuple[pd.DataFrame]:
        print("Loading dataset with version", version)
        return pd.read_csv(f"./data/instances{version}.csv", header=0), pd.read_csv(f"./data/labels{version}.csv", header=0)

    def convert_graph_list_to_dataframe(self, ds: List[Graph]) -> pd.DataFrame:
        """
        Converts the training data - a list of graph objects - to a pandas dataframe with the following columns:

            node_a_family ; node_b_family ; n_parts_in_graph ; n_parts_of_fam1 ; n_parts_of_fam2 ; ... ; n_parts_of_famK

        where K is the largest family id (currently set to 105 just to be sure).
        :param ds: the dataset
        :returns: the x instances and labels for all part pairs in each graph of ds
        """
        instances: List[List] = []
        labels: List[List[bool]] = []
        index = 0
        graphs = 1
        if ds is not None:
            for g in ds:
                print(graphs)
                graphs += 1
                edges = g.get_edges()
                for i, n1 in enumerate(g.get_nodes()):
                    for j, n2 in enumerate(g.get_nodes()):
                        if i == j:
                            continue
                        has_edge = n2 in edges[n1]
                        # There are way more "non-edges" than existing edges -> only take half of the non-edges
                        if has_edge or random() < 0.5:
                            # Creates a single row in the above-mentioned format
                            row = self.get_instance(n1.get_part(), n2.get_part(), g.get_parts())
                            instances.append(row)
                            labels.append([1 if has_edge else 0])
                            index += 1

        return pd.DataFrame(instances), pd.DataFrame(labels)

    def get_instance(self, p1: Part, p2: Part, parts: Set[Part]) -> [Tuple[int]]:
        """
        Returns a single row for two parts of a given set of parts in the following format:

            node_a_family ; node_b_family ; n_parts_in_graph ; n_parts_of_fam1 ; n_parts_of_fam2 ; ... ; n_parts_of_famK
        """
        family_occs = [0] * 105  # 105 different family_id values, in case their are more than in the training dataset
        for p in parts:
            index = int(self.part_to_index(p))
            family_occs[index] += 1

        front = [self.part_to_index(p1), self.part_to_index(p2), len(parts)]
        front.extend(family_occs)
        return front

# Different row formats have been tried. Using only the part pairs families and number of parts in the set
# resulted in a ~76% accuracy
    # def get_instance(self, p1: Part, p2: Part, parts: Set[Part]) -> [Tuple[int]]:
    #     return [self.part_to_index(p1), self.part_to_index(p2), len(parts)]

    def part_to_index(self, p: Part) -> int:
        """
        Returns the family Id of a part. Different representations of parts have been tried,
        for example, embeddings of encoders, but all yielded worse results than just the family alone.
        """
        return int(p.get_family_id())
