import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.nn.activations import LeakyReLU, Sigmoid, Softmax


class TestLeakyReLU(unittest.TestCase):
    def test_alpha_forward(self):
        x = Tensor([-2.0, 0.0, 3.0])
        activation = LeakyReLU(alpha=0.1)
        out = activation.forward(x)
        np.testing.assert_allclose(out.data, [-0.2, 0.0, 3.0])

    def test_default_alpha(self):
        x = Tensor([-2.0, 0.0, 3.0])
        activation = LeakyReLU()
        out = activation.forward(x)
        np.testing.assert_allclose(out.data, [-0.02, 0.0, 3.0])

    def test_invalid_tensor(self):
        activation = LeakyReLU()
        with self.assertRaises(TypeError):
            activation.forward([-2.0, 3.0])


class TestSigmoid(unittest.TestCase):
    def test_forward_and_backward(self):
        x = Tensor([-2.0, 0.0, 2.0])
        activation = Sigmoid()

        out = activation.forward(x)
        out.sum().backward()

        expected = 1.0 / (1.0 + np.exp(-x.data))
        np.testing.assert_allclose(out.data, expected)
        np.testing.assert_allclose(x.grad, expected * (1.0 - expected))

    def test_invalid_tensor(self):
        activation = Sigmoid()
        with self.assertRaises(TypeError):
            activation.forward([-2.0, 0.0, 2.0])


class TestSoftmax(unittest.TestCase):
    def test_forward_shape(self):
        softmax = Softmax()
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        probabilities = softmax.forward(x)
        self.assertEqual(probabilities.data.shape, (2, 3))

    def test_sum_p_equals_one(self):
        softmax = Softmax()
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        probabilities = softmax.forward(x)
        row_sum = probabilities.data.sum(axis=1)
        np.testing.assert_allclose(row_sum, [1.0, 1.0])

    def test_large_logits_are_finite(self):
        softmax = Softmax()
        x = Tensor([[1000, 1001, 1002]])

        probabilities = softmax.forward(x)
        result = np.isfinite(probabilities.data).all()

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
