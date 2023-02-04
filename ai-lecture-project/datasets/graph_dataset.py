import pickle
from typing import List

from torch.utils.data import Dataset

from config.config import root_path
from graph import Graph


class GraphDataset(Dataset):
    """
    This dataset provides lists of parts as inputs and full graphs as labels.
    """

    def __init__(self):
        with open(f"{root_path}/data/graphs.dat", 'rb') as file:
            self.graphs: List[Graph] = pickle.load(file)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph.get_parts(), graph