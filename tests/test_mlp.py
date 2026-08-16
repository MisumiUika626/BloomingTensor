import unittest

from src.autograd.tensor import Tensor
from src.models.mlp import MLP


class TestMLP(unittest.TestCase):
    def test_forward_shape(self):
        model = MLP([3, 4, 1])

        inputs = Tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
            ]
        )

        output = model.forward(inputs)

        self.assertEqual(output.data.shape, (2, 1))
