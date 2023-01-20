import torch
from torch import Tensor

from encoder.abstract_encoder import AbstractEncoder
from part import Part

# Known Id Range:
# Part id: 0 - 2270 (both inclusive)
# Family id: 0 - 100 (both inclusive)
PART_ID_BITS = 2271
FAMILY_ID_BITS = 101

class OneHotEncoder(AbstractEncoder):

    def get_encoding_size(self):
        return PART_ID_BITS + FAMILY_ID_BITS

    def encode(self, part: Part) -> Tensor:
        encoding = torch.zeros((PART_ID_BITS + FAMILY_ID_BITS))
        encoding[int(part.get_part_id())] = 1.0
        encoding[int(part.get_family_id()) + PART_ID_BITS] = 1.0
        return encoding

    def decode(self, part: Tensor) -> Part:
        maxActivationFamily = 0
        maxActivationPart = 0
        partId = 0
        familyIndex = PART_ID_BITS
        for i in range(PART_ID_BITS):
            if (part[i] > maxActivationPart):
                partId = i
                maxActivationPart = part[i]
        for i in range(PART_ID_BITS, PART_ID_BITS + FAMILY_ID_BITS):
            if (part[i] > maxActivationFamily):
                familyIndex = i
                maxActivationFamily = part[i]
        return Part(partId, familyIndex - PART_ID_BITS)