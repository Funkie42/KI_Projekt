
from torch import nn, Tensor

from encoder.auto_encoder_trainer import EncoderTrainingNetwork

import torch.nn.functional as F


class AutoEncoder():
    encode1: nn.Linear
    encode2: nn.Linear

    def __init__(self, encode1: nn.Linear, encode2: nn.Linear):
        self.encode1 = encode1
        self.encode2 = encode2

    def encodeTensor(self, tensor: Tensor) -> Tensor:
        out = F.relu(self.encode1(tensor.detach()))
        return F.relu(self.encode2(out)).detach()
