import numpy as np
from numpy import ndarray, float32
from torch import Tensor

from encoder.abstract_encoder import AbstractEncoder
from part import Part

# Known Id Range:
# Part id: 0 - 2270 (both inclusive)
# Family id: 0 - 100 (both inclusive)
PART_ID_BITS = 12
FAMILY_ID_BITS = 7

class BinaryEncoder(AbstractEncoder):
    "Encodes the part id and family id in little endian encoding"

    def get_encoding_size(self):
        return PART_ID_BITS + FAMILY_ID_BITS

    def encode(self, part: Part) -> Tensor:
        array = [0.0] * (PART_ID_BITS + FAMILY_ID_BITS)
        partId = int(part.get_part_id())
        familyId = int(part.get_family_id())
        for i in range(PART_ID_BITS + FAMILY_ID_BITS):
            if (i < PART_ID_BITS):
                array[i] = partId % 2
                partId //= 2
            else:
                array[i] = familyId % 2
                familyId //= 2
        return Tensor(array)

    def decode(self, part: Tensor) -> Part:
        bits = [0] * (PART_ID_BITS + FAMILY_ID_BITS)
        for i in range(PART_ID_BITS + FAMILY_ID_BITS):
            bits[i] = 1 if part[i] > 0.5 else 0

        partId = 0
        familyId = 0
        for i in range(PART_ID_BITS + FAMILY_ID_BITS - 1, -1, -1):
            if (i < PART_ID_BITS):
                partId *= 2
                partId += bits[i]
            else:
                familyId *= 2
                familyId += bits[i]
        return Part(partId, familyId)