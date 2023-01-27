import torch
from torch import nn
from torch.nn import LSTMCell


class LSTMGraphPredictor(nn.Module):

    def __init__(self, encoded_part_size: int):
        super().__init__()

        self.input_dim = encoded_part_size
        self.builder_hidden_dim = 1500
        self.construction_plan_size = self.builder_hidden_dim
        self.query_size = 300

        self.builder = LSTMCell(input_size=self.input_dim, hidden_size=self.builder_hidden_dim, bias=True)

    def inputToLSTMSequence(self, x):
        """
        Turns the given input into an LSTM sequence. This function thereby takes care of the fact that LSTMs require
        ordered input whereas the nodes in a graph have no natural ordering.
        To account for this, this function sorts the parts of each graph.

        This function returns a tensor with the following dimension:
        0: sequence
        1: batch
        2: part encoding
        """
        # Alternatively, this function may construct every permutation of part orders
        # TODO
        return torch.transpose(x, 0, 1)

    def makeConstructionPlan(self, x):
        """
        Creates a construction plan for the given input, which must be structured just like in ´forward`.
        """
        batch_size = len(x)
        lstm_input = self.inputToLSTMSequence(x)
        lstm_hidden_state = torch.zeros(batch_size, self.builder_hidden_dim)
        lstm_cell_state = torch.zeros(batch_size, self.builder_hidden_dim)

        # Iterate over sequence
        for elem in lstm_input:
            (lstm_hidden_state, lstm_cell_state) = self.builder(elem, (lstm_hidden_state, lstm_cell_state))

        return lstm_hidden_state

    def makeQuery(self, x, part1_index, part2_index):
        """
        Makes a query from the given input using the parts at the given indices. The input must have the same shape
        as required by `forward`.

        Returns a tensor of the following shape:
        Dimension 0: batch
        Dimension 1: query (size is always self.query_size)
        """
        part1 = x[:, part1_index] # For each batch, get the element at the part index
        part2 = x[:, part2_index]

        query_in = torch.cat((part1, part2), dim=1)
        query_out = query_in # TOOO Pass through fully connected network
        return query_out

    def forward(self, x):
        """
        Forwards the given input. The input dimensions are as follows:
        dimension 0: batch
        dimension 1: parts (sequence)
        dimension 2: encoded version of each part
        """
        construction_plan = self.makeConstructionPlan(x)


