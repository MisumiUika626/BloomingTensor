"""Tests for the Tensor-based neural-network Linear layer."""

import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.nn.linear import Linear


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.model = Linear(in_features=3, out_features=2)
        self.model.weight.data = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        self.model.bias.data = np.array([0.5, -0.5])

    def test_forward(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = self.model.forward(x)

        self.assertEqual(output.data.shape, (2, 2))
        np.testing.assert_allclose(
            output.data,
            [[22.5, 27.5], [49.5, 63.5]],
        )

    def test_rejects_non_tensor_input(self):
        with self.assertRaises(TypeError):
            self.model.forward([[1.0, 2.0, 3.0]])

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            self.model.forward(Tensor([1.0, 2.0, 3.0]))

    def test_rejects_wrong_feature_dimension(self):
        with self.assertRaises(ValueError):
            self.model.forward(Tensor([[1.0, 2.0]]))

    def test_backward(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss = self.model.forward(x).sum()
        loss.backward()

        np.testing.assert_allclose(
            self.model.weight.grad,
            [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]],
        )
        np.testing.assert_allclose(self.model.bias.grad, [2.0, 2.0])
        np.testing.assert_allclose(
            x.grad,
            [[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]],
        )

    def test_parameters(self):
        parameters = self.model.parameters()

        self.assertEqual(len(parameters), 2)
        self.assertIs(parameters[0], self.model.weight)
        self.assertIs(parameters[1], self.model.bias)


if __name__ == "__main__":
    unittest.main()
