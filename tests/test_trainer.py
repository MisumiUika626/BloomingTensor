import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.trainers.trainer import ToyTrainer


class TestToyTrainer(unittest.TestCase):
    def test_mse_loss(self):
        trainer = ToyTrainer()
        prediction = Tensor([[2.0], [4.0]])
        target = Tensor([[4.0], [8.0]])

        loss = trainer.compute_loss(prediction, target)

        self.assertEqual(loss.data.shape, ())
        np.testing.assert_allclose(loss.data, 10.0)

    def test_loss_backward(self):
        trainer = ToyTrainer()
        prediction = Tensor([[2.0], [4.0]])
        target = Tensor([[4.0], [8.0]])

        loss = trainer.compute_loss(prediction, target)
        loss.backward()

        # d(MSE)/d(prediction) = 2 * (prediction - target) / N
        np.testing.assert_allclose(
            prediction.grad,
            [[-2.0], [-4.0]],
        )


if __name__ == "__main__":
    unittest.main()
