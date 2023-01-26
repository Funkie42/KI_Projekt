import unittest

import torch

from encoder.binary_encoder import BinaryEncoder
from encoder.one_hot_encoder import OneHotEncoder
from part import Part


class TestEncoder(unittest.TestCase):

    def test_decodeself_binary(self):
        self.sut = BinaryEncoder()
        self.verifyEncodeSelf(Part(1, 4))
        self.verifyEncodeSelf(Part(17, 8))
        self.verifyEncodeSelf(Part(2174, 98))
        self.verifyEncodeSelf(Part(19, 3))

    def test_binary_encode(self):
        self.sut = BinaryEncoder()
        encoding = self.sut.encode(Part(13, 2))
        expectedEncoding = torch.Tensor([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 0, 0, 0, 0, 0])
        self.assertTrue((expectedEncoding == encoding).all())

    def test_decodeself_onehot(self):
        self.sut = OneHotEncoder()
        self.verifyEncodeSelf(Part(21, 4))
        self.verifyEncodeSelf(Part(17, 99))
        self.verifyEncodeSelf(Part(2201, 1))
        self.verifyEncodeSelf(Part(158, 23))

    def verifyEncodeSelf(self, part: Part):
        encoding = self.sut.encode(part)
        oPart = self.sut.decode(encoding)
        self.assertEqual(part.get_part_id(), oPart.get_part_id())
        self.assertEqual(part.get_family_id(), oPart.get_family_id())