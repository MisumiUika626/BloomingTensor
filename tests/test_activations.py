import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.nn.activations import LeakyReLU


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


if __name__ == "__main__":
    unittest.main()
