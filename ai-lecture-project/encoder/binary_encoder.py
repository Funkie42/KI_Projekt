import numpy as np
from numpy import ndarray

from encoder.abstract_encoder import AbstractEncoder
from part import Part


PART_ID_BITS = 11
FAMILY_ID_BITS = 5

class BinaryEncoder(AbstractEncoder):
    "Encodes the part id and family id in little endian encoding"

    def encode(self, part: Part) -> ndarray:
        array = [0.0] * (PART_ID_BITS + FAMILY_ID_BITS)
        partId = part.get_part_id()
        familyId = part.get_family_id()
        for i in range(PART_ID_BITS + FAMILY_ID_BITS):
            if (i < PART_ID_BITS):
                array[i] = partId % 2
                partId //= 2
            else:
                array[i] = familyId % 2
                familyId //= 2
        return np.array(array, dtype=float)

    def decode(self, part: ndarray) -> Part:
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