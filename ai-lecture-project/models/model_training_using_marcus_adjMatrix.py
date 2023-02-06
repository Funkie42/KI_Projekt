from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim, Tensor
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR

import ffnn.feedforward_neural_network_model as ffnn
import encoder.auto_encoder_trainer as encoder_trainer
import encoder.auto_encoder_decoder as auto_encoder
from config.config import root_path
from datasets.edge_vector_dataset import EdgeVectorDataset

from torch.utils.data import random_split, DataLoader
from graph import Graph
from lstm.lstm_network import collate_encoded_parts_list, predict, get_cuda_memory_info
from node import Node
from part import Part

encoder = auto_encoder.loadPretrainedAutoEncoder()

input_dim = encoder.get_encoding_size()
hidden_dim_1 = 300
hidden_dim_2 = 60
hidden_dim_3 = 10
output_dim = 1

# Only what nodes connect to what. Not what parts these nodes actually have
base_dataset = EdgeVectorDataset(part_encoder=encoder)

(train_data, val_data, test_data) = random_split(base_dataset, [0.7, 0.15, 0.15],
                                                     generator=torch.Generator().manual_seed(7))
torch.manual_seed(42)


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

print("Data prepared and Nodes encoded.")

model = ffnn.FeedforwardNeuralNetworkModel(input_dim, output_dim, hidden_dim_2)
criterion = nn.CrossEntropyLoss() #nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
lr_scheduler = MultiStepLR(optimizer, milestones=[10, 30, 80, 150, 250], gamma=0.2)

data_per_epoch = len(train_data_loader)
val_data_per_epoch = len(val_data_loader)
val_iterator = iter(val_data)
loss_per_epoch = []
val_loss = []

n_epochs = 4

print("Start training")

for epoch in range(n_epochs):
    loss_per_epoch.append(
        predict(
            model,
            train_data_loader,
            lossCriterion=criterion,
            optimizer=optimizer,
            train=True,
            message=f"Training underway. Epoch {epoch + 1}/{n_epochs} batch {{}} (CUDA memory usage: {get_cuda_memory_info()})"
        )
    )

    val_loss.append(
        predict(
            model,
            val_data_loader,
            lossCriterion=criterion,
            optimizer=None,
            train=False,
            message=f"Validation underway. Epoch {epoch + 1}/{n_epochs} batch {{}} (CUDA memory usage: {get_cuda_memory_info()})"
        )
    )
    lr_scheduler.step()
print()
torch.save(model.state_dict(), f'{root_path}/data/trained_lstm.dat')

x_epochs = list(range(1, n_epochs + 1))
plt.plot(x_epochs, loss_per_epoch, color='blue', label='Training loss')
plt.plot(x_epochs, val_loss, color='red', label='Validation loss')
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



