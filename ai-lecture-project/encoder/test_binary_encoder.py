import unittest

import numpy as np

from encoder.binary_encoder import BinaryEncoder
from part import Part


class TestBinaryEncoder(unittest.TestCase):

    def setUp(self):
        self.sut = BinaryEncoder()

    def test_decodeself(self):
        self.verifyEncodeSelf(Part(1, 4))
        self.verifyEncodeSelf(Part(17, 8))
        self.verifyEncodeSelf(Part(32, 1))
        self.verifyEncodeSelf(Part(19, 3))

    def test_encode(self):
        encoding = self.sut.encode(Part(13, 2))
        expectedEncoding = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 0, 0, 0]
        self.assertTrue((np.array(expectedEncoding) == encoding).all())

    def verifyEncodeSelf(self, part: Part):
        encoding = self.sut.encode(part)
        oPart = self.sut.decode(encoding)
        self.assertEqual(part.get_part_id(), oPart.get_part_id())
        self.assertEqual(part.get_family_id(), oPart.get_family_id())