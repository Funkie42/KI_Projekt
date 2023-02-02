import math

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from torch.nn import LSTMCell
from torch.utils.data import random_split, DataLoader

from config.config import device
from datasets.edge_vector_dataset import EdgeVectorDataset
from encoder.auto_encoder_decoder import loadPretrainedAutoEncoder
from lstm.positional_encoding import PositionalEncoding


class LSTMGraphPredictor(nn.Module):

    def __init__(self, encoded_part_size: int):
        super().__init__()

        self.input_dim = encoded_part_size
        self.builder_hidden_dim = 1500
        self.construction_plan_size = self.builder_hidden_dim
        self.builder = LSTMCell(input_size=self.input_dim, hidden_size=self.builder_hidden_dim, bias=True)

        self.query_size = 300
        self.query_in_size = 2 * encoded_part_size
        self.query_pos_encoder = PositionalEncoding(d_model=self.query_in_size)
        self.query_fc1 = nn.Linear(self.query_in_size, 600)
        self.query_fc2 = nn.Linear(600, self.query_size)

        self.evaluator_fc1 = nn.Linear(self.construction_plan_size + self.query_size, 1000)
        self.evaluator_norm1 = nn.BatchNorm1d(num_features=1000)
        self.evaluator_fc2 = nn.Linear(1000, 100)
        self.evaluator_norm2 = nn.BatchNorm1d(num_features=100)
        self.evaluator_fc3 = nn.Linear(100, 10)
        self.evaluator_fc4 = nn.Linear(10, 1)

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
        # First sort each sequence, then swap batch and sequence ordering
        return torch.sort(x, dim=1)[0].transpose(0, 1)

    def makeConstructionPlan(self, x):
        """
        Creates a construction plan for the given input, which must be structured just like in ´forward`.
        Returns a construction plan of shape [batch_size, construction_plan_size].
        """
        batch_size = len(x)
        lstm_input = self.inputToLSTMSequence(x)
        lstm_hidden_state = torch.zeros(batch_size, self.builder_hidden_dim).to(device)
        lstm_cell_state = torch.zeros(batch_size, self.builder_hidden_dim).to(device)

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

        # Concatenate parts
        raw_parts = torch.cat((part1, part2), dim=1)

        # Add positional encoding
        positionally_encoded_parts = self.query_pos_encoder(raw_parts.unsqueeze(0)).squeeze(0)

        # Pass through FCN
        query_out = F.relu(self.query_fc1(positionally_encoded_parts))
        query_out = F.relu(self.query_fc2(query_out))

        return query_out

    def evaluateQuery(self, constructionPlan: Tensor, query: Tensor) -> Tensor:
        """
        Evaluates the given query on the given construction plan. Returns the predicted likelyhood that the parts
        given through the query are connected in the given construction plan.
        The construction plan must be of shape [batch_size, construction_plan_size].
        The query must be of shape [batch_size, query_size].
        Returns a tensor of shape [batch_size].
        """
        evaluator_input = torch.cat((constructionPlan, query), dim=1)

        eval = self.evaluator_fc1(evaluator_input)
        #eval = self.evaluator_norm1(eval)
        eval = self.evaluator_fc2(eval)
        #eval = self.evaluator_norm2(eval)
        eval = self.evaluator_fc3(eval)
        eval = F.relu(self.evaluator_fc4(eval))

        return torch.squeeze(eval, 1) # Turn output per batch from 1-element-vector into scalar

    def forward(self, x):
        """
        Forwards the given input. The input dimensions are as follows:
        dimension 0: batch (can be omitted)
        dimension 1: parts (sequence)
        dimension 2: encoded version of each part

        Returns a tensor of the following shape
        [batch_size (missing if omitted in input), edge_vector_size]

        The edge_vector size is (sequence_size ** 2 - sequence_size) / 2. It represents one half of the adjacency matrix.
        (The first element of the edge vector represents the edge between part 1 and part 2, next between part 1 and
        part 3, next between part 1 and part 4, ..., part 1 and part n, part 2 and part 3, part 2 and part 4, ...
        Basically the same as if you read the adjacency matrix from the top left, but only considered the upper right
        half.)
        """
        unsqueezed = False
        if len(x.shape) == 2:
            unsqueezed = True
            x = torch.unsqueeze(x, 0)

        amount_parts = x.shape[1]
        batch_size = x.shape[0]
        edge_vector_size = self.calcEdgeVectorSizeFromAdjMatrixAxisSize(amount_parts)

        construction_plan = self.makeConstructionPlan(x)

        query_list = []
        for part1Index in range(amount_parts - 1):
            for part2Index in range(part1Index+1, amount_parts):
                query_list.append(self.makeQuery(x, part1Index, part2Index))

        # We want to execute all queries at once, so we intertwine them with the batches.

        # [query_sequence_length, batch_size, query_size]
        query_tensor = torch.stack(query_list)

        # [query_sequence_length * batch_size, query_size]
        # First dimension ordering is: B1Q1, B2Q1, B3Q1, ..., B1Q2, B2Q2, B3Q2
        multi_query = query_tensor.reshape(-1, self.query_size)

        # Query sequence length is the same as edge vector length.
        # [edge_vector_length * batch_size]
        multi_construction_plan = construction_plan.repeat(edge_vector_size, 1)
        multi_result = self.evaluateQuery(multi_construction_plan, multi_query)

        # [edge_vector_length, batch_size]
        swapped_result = multi_result.reshape(edge_vector_size, batch_size)

        # [batch_size, edge_vector_length]
        result = swapped_result.transpose(0, 1)

        if unsqueezed:
            result = torch.squeeze(result)

        return result

    def calcEdgeVectorSizeFromAdjMatrixAxisSize(self, adjMatrixAxisSize: int) -> int:
        return (adjMatrixAxisSize ** 2 - adjMatrixAxisSize) // 2

    def calcAdjMatrixAxisSizeFromEdgeVectorSize(self, edgeVectorSize: int) -> int:
        return math.isqrt(2 * edgeVectorSize) + 1

    def edgeVectorToAdjMatrix(self, edge_vector: Tensor) -> Tensor:
        """
        Converts an edge vector tensor to an adjacency matrix.
        The input must be of shape [batch_size (can be omitted), edge_vector_size].
        """
        unsqueezed = False
        if len(edge_vector.shape) == 1:
            edge_vector = edge_vector.unsqueeze(0)

        edge_vector_size = edge_vector.shape[1]
        axis_size = self.calcAdjMatrixAxisSizeFromEdgeVectorSize(edge_vector_size)
        adj = torch.zeros((edge_vector.shape[0], axis_size, axis_size))
        index = 0
        for i1 in range(axis_size-1):
            for i2 in range(i1 + 1, axis_size):
                edges = edge_vector[:, index]
                adj[:, i1, i2] = edges
                adj[:, i2, i1] = edges
                index += 1

        if unsqueezed:
            adj = adj.squeeze(0)

        return adj

    def adjMatrixToEdgeVector(self, adj_matrix: Tensor) -> Tensor:
        """
        Turns the given adjacency matrix into an edge vector. You may also pass a batch of adjacency matrices
        to get a batch of edge vectors.
        """
        unsqueezed = False
        if len(adj_matrix.shape) < 3:
            adj_matrix = adj_matrix.unsqueeze(0)

        axis_size = adj_matrix.shape[2]

        if axis_size != adj_matrix.shape[1]:
            raise RuntimeError("The given adjacency matrix's dimensions are not of the same size! How can this be an adjacency matrix??")

        edge_vector = []
        for i1 in range(axis_size - 1):
            for i2 in range(i1 + 1, axis_size):
                edge_vector.append(adj_matrix[:, i1, i2])

        edge_vector = torch.stack(edge_vector).T

        if unsqueezed:
            edge_vector = edge_vector.squeeze(0)
        return edge_vector

if __name__ == '__main__':

    encoder = loadPretrainedAutoEncoder()
    base_dataset = EdgeVectorDataset(part_encoder=encoder)

    (train_data, val_data, test_data) = random_split(base_dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(7))

    torch.manual_seed(7)
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    network = LSTMGraphPredictor(encoder.get_encoding_size()).to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.5)
    lossCriterion = nn.MSELoss()
    loss_per_step = []
    val_loss = []

    data_sets = len(train_data_loader)
    val_iterator = iter(val_data)

    for (index, (parts, label)) in enumerate(train_data_loader):
        for i in range(500):
            optimizer.zero_grad()
            prediction = network(parts)
            loss = lossCriterion(prediction, label)
            loss_per_step.append(float(loss))
            #print(f"Before backward: {network.query_fc1.weight.grad}")
            loss.backward()
            #print(f"After backward: {network.query_fc1.weight.grad}")
            optimizer.step()
        #print(f"\rExecuted training step {index}/{data_sets}. Loss={float(loss)}", end="")

        break
        if index % 10 == 0:
            network.eval()
            (parts, label) = next(val_iterator)
            prediction = network(parts)
            loss = lossCriterion(prediction, label)
            val_loss.append(float(loss))
            network.train()

    x_loss = list(range(1, len(loss_per_step)+1))
    x_val = list(range(1, 10*len(val_loss)+1, 10))
    plt.plot(x_loss, loss_per_step, color='blue', label='Training loss')
    plt.plot(x_val, val_loss, color='red', label='Validation loss')
    plt.title("LSTM Network loss")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.ylim(0, 0.5)
    plt.show()

    print()