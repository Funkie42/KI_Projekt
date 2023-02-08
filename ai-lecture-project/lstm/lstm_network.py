import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor, BoolTensor
from torch.nn import LSTMCell
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split, DataLoader

from config.config import device, root_path
from datasets.edge_vector_dataset import EdgeVectorDataset
from encoder.one_hot_encoder import OneHotEncoder
from lstm.positional_encoding import PositionalEncoding


class LSTMGraphPredictor(nn.Module):

    def __init__(self, encoded_part_size: int):
        super().__init__()

        self.part_encoding_size = encoded_part_size

        self.input_dim = encoded_part_size
        self.builder_hidden_dim = 600
        self.construction_plan_size = self.builder_hidden_dim
        self.builder = LSTMCell(input_size=self.input_dim, hidden_size=self.builder_hidden_dim, bias=True)

        self.query_size = 200
        self.query_in_size = 2 * encoded_part_size
        self.query_pos_encoder = PositionalEncoding(d_model=self.part_encoding_size)
        self.query_fc1 = nn.Linear(self.query_in_size, 600)
        self.query_fc2 = nn.Linear(600, self.query_size)
        self.query_norm = nn.BatchNorm1d(num_features=self.query_size)

        evaluator_l1_size = self.construction_plan_size + self.query_size
        evaluator_l2_size = 300
        evaluator_l3_size = 75
        self.evaluator_fc1 = nn.Linear(evaluator_l1_size, evaluator_l2_size)
        self.evaluator_norm1 = nn.BatchNorm1d(num_features=evaluator_l2_size)
        self.evaluator_fc2 = nn.Linear(evaluator_l2_size, evaluator_l3_size)
        self.evaluator_norm2 = nn.BatchNorm1d(num_features=evaluator_l3_size)
        self.evaluator_fc3 = nn.Linear(evaluator_l3_size, 1)

        # He Initialization for all layers using ReLU
        torch.nn.init.kaiming_uniform_(self.query_fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.query_fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.evaluator_fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.evaluator_fc2.weight, nonlinearity='relu')

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

        return lstm_cell_state

    def makeEncodedPartPair(self, x, part1_index, part2_index):
        """
        Constructs the input for the query creator from the given input at the given indices. The input must have the
        shape [sequence_len, batch_size, part_encoding_size]

        Returns a tensor of size [batch_size, encoding_size*2]. The first dimension is the batch, the second dimension
        the input to the query creator.
        """
        part1 = x[part1_index]  # For each batch, get the element at the part index
        part2 = x[part2_index]

        # Concatenate parts
        return torch.cat((part1, part2), dim=1)

    def makeQuery(self, prepared_queries):
        """
        Makes a query from the given two encoded parts. I.e., the output of makeEncodedPartPair.
        The input must be of shape [batch_size, 2*encoding_size].

        Returns a tensor of the following shape:
        Dimension 0: batch
        Dimension 1: query (size is always self.query_size)
        """
        # Pass through FCN
        query_out = F.relu(self.query_fc1(prepared_queries))
        query_out = F.relu(self.query_fc2(query_out))
        # Not using ReLU here, because it would screw over the sigmoid loss for probability calculation
        query_out = self.query_norm(query_out)

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

        eval = F.relu(self.evaluator_fc1(evaluator_input))
        eval = self.evaluator_norm1(eval)
        eval = F.relu(self.evaluator_fc2(eval))
        eval = self.evaluator_norm2(eval)
        eval = self.evaluator_fc3(eval)
        # eval = self.evaluator_fc4(eval)

        return torch.squeeze(eval, 1)  # Turn output per batch from 1-element-vector into scalar

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
        edge_vector_size = calcEdgeVectorSizeFromAdjMatrixAxisSize(amount_parts)

        construction_plan = self.makeConstructionPlan(x)

        positionally_encoded_parts = x.transpose(0, 1)  # swap batch and sequence
        positionally_encoded_parts = self.query_pos_encoder(positionally_encoded_parts)

        batched_part_pairs = []
        for part1Index in range(amount_parts - 1):
            for part2Index in range(part1Index + 1, amount_parts):
                batched_part_pairs.append(self.makeEncodedPartPair(positionally_encoded_parts, part1Index, part2Index))

        batched_part_pairs = torch.stack(batched_part_pairs)

        # We want to make and execute all queries at once, so we intertwine them with the batches.
        # Currently prepared_batched_queries contains a sequence of batched queries:
        # [sequence_length, batch_size, 2*part_encoding_size]
        # Now we want to merge the sequence length and batch size dimension, since we can execute all queries of each
        # batch in parallel as well.
        # Ordering is S1B1, S1B2, S1B3, ..., S2B1, S2B2, ...
        merged_batched_part_pairs = batched_part_pairs.reshape(-1, 2 * self.part_encoding_size)

        # Now, pass all of these queries through the query maker
        # Output shape is:
        # [sequence_length * batch_size, query_size]
        multi_queries = self.makeQuery(merged_batched_part_pairs)

        # Query sequence length is the same as edge vector length.
        # [edge_vector_length * batch_size]
        multi_construction_plan = construction_plan.repeat(edge_vector_size, 1)
        multi_result = self.evaluateQuery(multi_construction_plan, multi_queries)

        # [edge_vector_length, batch_size]
        swapped_result = multi_result.reshape(edge_vector_size, batch_size)

        # [batch_size, edge_vector_length]
        result = swapped_result.transpose(0, 1)

        if unsqueezed:
            result = torch.squeeze(result)

        return result

def calcEdgeVectorSizeFromAdjMatrixAxisSize(adjMatrixAxisSize: int) -> int:
    return (adjMatrixAxisSize ** 2 - adjMatrixAxisSize) // 2


def calcAdjMatrixAxisSizeFromEdgeVectorSize(edgeVectorSize: int) -> int:
    return math.isqrt(2 * edgeVectorSize) + 1

def edgeVectorToAdjMatrix(edge_vector: Tensor) -> Tensor:
    """
    Converts an edge vector tensor to an adjacency matrix.
    The input must be of shape [batch_size (can be omitted), edge_vector_size].
    """
    unsqueezed = False
    if len(edge_vector.shape) == 1:
        edge_vector = edge_vector.unsqueeze(0)
        unsqueezed = True

    edge_vector_size = edge_vector.shape[1]
    axis_size = calcAdjMatrixAxisSizeFromEdgeVectorSize(edge_vector_size)
    adj = torch.zeros((edge_vector.shape[0], axis_size, axis_size))
    index = 0
    for i1 in range(axis_size - 1):
        for i2 in range(i1 + 1, axis_size):
            edges = edge_vector[:, index]
            adj[:, i1, i2] = edges
            adj[:, i2, i1] = edges
            index += 1

    if unsqueezed:
        adj = adj.squeeze(0)

    return adj

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


def collate_encoded_parts_list(graphs_and_labels: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, BoolTensor]]:
    """
    Collate function for edge_vector labelled and part-encoded graphs. The first element of each tuple is the parts
    entry, the second element the label (edge vector). Each parts entry must be a two-dimensional tensor where
    the first dimension is the parts sequence and the second dimension is the encoding for each part.
    All elements must have the same encoding size.

    Returns a 2-tuple. The first element is the tensor containing a batch input, the second element is the labelling.
    The labelling is a tensor together with a binary mask.
    """
    encoding_size = len(graphs_and_labels[0][0][0])
    max_amount_parts = 0
    max_edge_vector_size = 0
    for (g, l) in graphs_and_labels:
        if len(g) > max_amount_parts:
            max_amount_parts = len(g)
            max_edge_vector_size = len(l)

    result_graphs = []
    result_labels = []
    result_masks = []
    for (g, l) in graphs_and_labels:
        result_graph = g
        result_label = l
        result_mask = torch.ones(len(l), dtype=torch.bool).to(g.device)
        length = len(g)
        if length < max_amount_parts:
            zero_parts = torch.zeros((max_amount_parts - length, encoding_size)).to(g.device)
            result_graph = torch.cat((zero_parts, g), dim=0)
            zero_labels = torch.zeros(max_edge_vector_size - len(l)).to(g.device)
            zero_mask = torch.zeros_like(zero_labels, dtype=torch.bool).to(g.device)
            # Since we prepended the "null" parts, the edge vector's first elements will refer to those. Thus,
            # we can also set all of them to zero.
            result_label = torch.cat((zero_labels, l), dim=0)
            result_mask = torch.cat((zero_mask, result_mask), dim=0)
        result_graphs.append(result_graph)
        result_labels.append(result_label)
        result_masks.append(result_mask)
    return torch.stack(result_graphs), (torch.stack(result_labels), torch.stack(result_masks).type(torch.bool))

def get_gradient_size(module) -> float:
    return float(module.weight.grad.pow(2).sum())


def predict(network: nn.Module, data_loader: DataLoader, lossCriterion, optimizer: Optimizer = None, train=True,
            message="batch {}") -> float:
    network.train(train)
    data_len = len(data_loader)
    loss_sum = 0.0
    for (index, (parts, label)) in enumerate(data_loader):
        print((f"\r{message}").format(f"{index}/{data_len}"), end="")

        if train:
            optimizer.zero_grad()

        prediction = network(parts)
        loss = lossCriterion(prediction, label)
        loss_sum += float(loss)

        if train:
            loss.backward()
            optimizer.step()

    return loss_sum / data_len

def get_cuda_memory_info():
    if (device == 'cpu'):
        return "-"
    t = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2))
    r = round(torch.cuda.memory_reserved(0) / (1024.0 ** 2))
    a = round(torch.cuda.memory_allocated(0) / (1024.0 ** 2))
    return f"{a}/{r}/{t} MiB"

def custom_loss(input: Tensor, target: Tuple[Tensor, BoolTensor]) -> Tensor:
    false_negative_penalty = 25 # False negatives are penalized additionally by this factor
    (target_tensor, target_mask) = target
    elem_len = target_tensor.shape[-1]
    loss_tensor = F.binary_cross_entropy_with_logits(
        input,
        target_tensor,
        weight=None,
        pos_weight=(torch.ones(elem_len) * false_negative_penalty).to(input.device),
        reduction='none',
    )
    # Ignore parts of prediction which come from padding - target_mask tells us which ones to consider
    masked_loss = torch.masked_select(loss_tensor, target_mask)
    return torch.mean(masked_loss)
    # unreduced_loss = -1 * (false_negative_penalty * (target * torch.log(prediction)) + ((1 - target) * torch.log(1 - prediction)))
    # unreduced_clamped_loss = torch.clamp(unreduced_loss, min=-100, max=100)
    # result = torch.mean(unreduced_clamped_loss)
    # if float(result) is float('nan'):
    #     print("Well damn.")
    # return result


if __name__ == '__main__':

    print("Loading encoder...")
    encoder = OneHotEncoder()

    print("Preparing data...")
    base_dataset = EdgeVectorDataset(part_encoder=encoder)
    (train_data, val_data, test_data) = random_split(base_dataset, [0.99, 0.005, 0.005],
                                                     generator=torch.Generator().manual_seed(7))

    torch.manual_seed(7)
    train_data_loader = DataLoader(
        train_data,
        batch_size=512,
        collate_fn=collate_encoded_parts_list,
        shuffle=True,
    )
    val_data_loader = DataLoader(
        val_data,
        batch_size=512,
        collate_fn=collate_encoded_parts_list,
        shuffle=False,
    )

    print("Initializing network...")
    network = LSTMGraphPredictor(encoder.get_encoding_size()).to(device)

    # network.load_state_dict(torch.load(f"{root_path}/data/trained_lstm_500_epochs.dat"))

    optimizer = optim.Adam(network.parameters(), lr=0.1)
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 30, 80, 150, 250], gamma=0.2)
    lossCriterion = custom_loss #nn.BCELoss()
    loss_per_epoch = []
    val_loss = []
    q1_weight_size_per_epoch = []
    e3_weight_size_per_epoch = []

    data_per_epoch = len(train_data_loader)
    val_data_per_epoch = len(val_data_loader)
    val_iterator = iter(val_data)

    (test_g, test_l) = test_data[1]
    print(f"Sample label:\n {test_l}")

    epochs = 4

    print("Starting training...")
    for epoch in range(epochs):
        sample_pred = network(test_g)
        sample_loss = lossCriterion(sample_pred, (test_l, torch.ones_like(test_l, dtype=torch.bool).type(torch.bool)))
        print(f"{sample_pred}: {sample_loss}")

        loss_per_epoch.append(
            predict(
                network,
                train_data_loader,
                lossCriterion=lossCriterion,
                optimizer=optimizer,
                train=True,
                message=f"Training underway. Epoch {epoch + 1}/{epochs} batch {{}} (CUDA memory usage: {get_cuda_memory_info()})"
            )
        )

        val_loss.append(
            predict(
                network,
                val_data_loader,
                lossCriterion=lossCriterion,
                optimizer=None,
                train=False,
                message=f"Validation underway. Epoch {epoch + 1}/{epochs} batch {{}} (CUDA memory usage: {get_cuda_memory_info()})"
            )
        )

        q1_weight_size_per_epoch.append(float(network.query_fc1.weight.abs().mean().to("cpu")))
        e3_weight_size_per_epoch.append(float(network.evaluator_fc3.weight.abs().mean().to("cpu")))

        lr_scheduler.step()
    print()

    torch.save(network.state_dict(), f'{root_path}/data/trained_lstm.dat')

    x_epochs = list(range(1, epochs + 1))
    plt.plot(x_epochs, loss_per_epoch, color='blue', label='Training loss')
    plt.plot(x_epochs, val_loss, color='red', label='Validation loss')
    plt.plot(x_epochs, q1_weight_size_per_epoch, color='orange', label='Query L1 weight size')
    plt.plot(x_epochs, e3_weight_size_per_epoch, color='violet', label='Evaluator L3 weight size')
    plt.title("LSTM Network loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    # plt.ylim(0, 1)
    plt.show()

    # plt.plot(x_loss, grad_qf1, color='red', label='Gradient query fc1')
    # plt.plot(x_loss, grad_qe1, color='blue', label='Gradient evaluator fc1')
    # plt.plot(x_loss, grad_qe4, color='green', label='Gradient evaluator fc4')
    #
    # plt.title("LSTM Gradient sizes")
    # plt.xlabel("Generation")
    # plt.ylabel("Gradient length")
    # plt.legend(loc="lower left")
    # plt.ylim(0, 0.001)
    # plt.show()

    print()
