from torch import nn, Tensor

import torch.nn.functional as F


class AutoDecoder():
    decode1: nn.Linear
    decode2: nn.Linear

    def __init__(self, decode1: nn.Linear, decode2: nn.Linear):
        self.decode1 = decode1
        self.decode2 = decode2

    def decodeTensor(self, tensor: Tensor) -> Tensor:
        out = F.relu(self.decode1(tensor.detach()))
        return F.relu(self.decode2(out)).detach()
