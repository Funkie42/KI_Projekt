from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from part import Part


class AbstractEncoder(ABC):

    @abstractmethod
    def get_encoding_size(self):
        """
        Returns the amount of elements an array encoding a part contains.
        """
        pass

    @abstractmethod
    def encode(self, part: Part) -> ndarray:
        """
        Encodes the given part in the corresponding encoding.
        This should always return a one-dimensional array or a scalar!
        """
        pass

    @abstractmethod
    def decode(self, part: ndarray) -> Part:
        """
        Decodes the given encoded part.
        """
        pass