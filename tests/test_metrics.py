import unittest

from src.autograd.tensor import Tensor
from src.metrics import accuracy


class TestAccuracy(unittest.TestCase):
    def setUp(self):
        self.logits = Tensor([[0.1, 2.0, 0.3], [3.0, 1.0, 0.0]])

    def test_all_predictions_correct(self):
        self.assertEqual(accuracy(self.logits, [1, 0]), 1.0)

    def test_half_predictions_correct(self):
        self.assertEqual(accuracy(self.logits, [1, 2]), 0.5)

    def test_all_predictions_wrong(self):
        self.assertEqual(accuracy(self.logits, [0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
