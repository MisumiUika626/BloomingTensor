import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.config import (
    RANDOM_SEED,
)
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

    def test_parameters(self):
        model = MLP([3, 4, 1])
        parameters = model.parameters()
        self.assertAlmostEqual(len(parameters), 4)

    def test_seeds(self):
        np.random.seed(RANDOM_SEED)
        model_a = MLP([3, 4, 1])
        np.random.seed(RANDOM_SEED)
        model_b = MLP([3, 4, 1])
        parameters_a = model_a.parameters()
        parameters_b = model_b.parameters()
        for parameter_a, parameter_b in zip(parameters_a, parameters_b):
            np.testing.assert_array_equal(parameter_a.data, parameter_b.data)
