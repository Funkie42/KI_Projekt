from torch import nn, Tensor

from encoder.abstract_encoder import AbstractEncoder
from encoder.auto_encoder_trainer import EncoderTrainingNetwork
from part import Part


class AutoEncoder(AbstractEncoder):

    encoder: AbstractEncoder
    encode1: nn.Linear
    encode2: nn.Linear
    decode1: nn.Linear
    decode2: nn.Linear

    def __init__(self, encoder: AbstractEncoder, trainingNetwork: EncoderTrainingNetwork):
        self.encoder = encoder
        self.encode1 = trainingNetwork.fc1
        self.encode2 = trainingNetwork.fc2
        self.decode1 = trainingNetwork.fc3
        self.decode2 = trainingNetwork.fc4

    def get_encoding_size(self):
        return self.encode2.out_features
        pass

    def encode(self, part: Part) -> Tensor:
        encoded_part = self.encoder.encode(part)
        encoded_part.requires_grad = False
        out = self.encode1(encoded_part)
        return self.encode2(out)

    def decode(self, part: Tensor) -> Part:
        tmp_grad = part.requires_grad
        part.requires_grad = False
        out = self.decode1(part)
        encoded_part = self.decode2(out)
        result = self.encoder.decode(encoded_part)
        part.requires_grad = tmp_grad
        return result
