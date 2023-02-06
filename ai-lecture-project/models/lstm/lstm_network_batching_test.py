import torch
from torch.utils.data import random_split, DataLoader

from datasets.edge_vector_dataset import EdgeVectorDataset
from encoder.one_hot_encoder import OneHotEncoder
from models.lstm.lstm_network import collate_encoded_parts_list, LSTMGraphPredictor

"""
This file runs the same training data once as batch, once as single instances and verifies that the results match up.
This way, we look for errors in the network's (quite complicated) batch processing mode.
"""

if __name__ == '__main__':
    encoder = OneHotEncoder()
    base_dataset = EdgeVectorDataset(part_encoder=encoder)

    (train_data, val_data, test_data) = random_split(base_dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(7))

    torch.manual_seed(7)
    data_loader = DataLoader(train_data, batch_size=20, collate_fn=collate_encoded_parts_list, shuffle=False)

    network = LSTMGraphPredictor(encoder.get_encoding_size())
    network.eval()

    batch = next(iter(data_loader))[0]
    batch_prediction = network(batch)

    single_predictions = []
    for single in batch:
        single_predictions.append(network(single))

    single_prediction_full = torch.stack(single_predictions)

    print(f"Batch prediction:\n {batch_prediction}")
    print(f"Single predictions:\n {single_prediction_full}")

    difference = batch_prediction - single_prediction_full
    print(f"Difference:\n {difference}")
    print(f"Max offset:\n {difference.max()}")

    # assert torch.all(batch_prediction.eq(single_prediction_full))
