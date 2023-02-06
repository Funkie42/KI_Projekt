import math

import torch
from torch import nn, Tensor

from encoder.one_hot_encoder import OneHotEncoder
from part import Part


# Not my code!!
# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# It's the same as in the lecture slides, as I've realized later on.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':
    p1 = Part(1, 2)
    p2 = Part(1, 3)
    p3 = Part(1, 2)
    p4 = Part(1, 3)

    encoder = OneHotEncoder()
    tensor = torch.stack((encoder.encode(p1), encoder.encode(p2), encoder.encode(p3), encoder.encode(p4)))

    print(tensor)

    encoding = PositionalEncoding(encoder.get_encoding_size())
    tensor2 = encoding(tensor)
    print(tensor2)