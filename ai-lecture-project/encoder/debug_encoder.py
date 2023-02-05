import torch
from torch import Tensor

from encoder.abstract_encoder import AbstractEncoder
from part import Part

class DebugEncoder(AbstractEncoder):
    """
    A very simple encoder which makes it easy to trace parts for debugging. Do not use in production,
    this encoder is an awful idea.
    Parts are encoded as 2-element tensor with the first element being the part id and the second element being
    the family id.
    """

    def get_encoding_size(self):
        return 2

    def encode(self, part: Part) -> Tensor:
        encoding = torch.zeros(2)
        encoding[0] = int(part.get_part_id())
        encoding[1] = int(part.get_family_id())
        return encoding

    def decode(self, part: Tensor) -> Part:
        return Part(part[0], part[1])