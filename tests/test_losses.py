import unittest

import numpy as np

from src.autograd.tensor import Tensor
from src.nn.losses import CrossEntropyLoss


class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward_returns_expected_scalar_loss(self):
        criterion = CrossEntropyLoss()
        logits = Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

        loss = criterion.forward(logits, [0, 2])

        self.assertEqual(loss.data.shape, ())
        np.testing.assert_allclose(loss.data, 0.4076059644443804)

    def test_large_logits_produce_finite_loss(self):
        criterion = CrossEntropyLoss()
        logits = Tensor(
            [[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -999.0]]
        )

        loss = criterion.forward(logits, [2, 0])

        self.assertTrue(np.isfinite(loss.data).all())

    def test_backward_matches_softmax_minus_one_hot_over_batch(self):
        criterion = CrossEntropyLoss()
        logits = Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        targets = np.array([0, 2])

        loss = criterion.forward(logits, targets)
        loss.backward()

        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(
            np.exp(shifted), axis=1, keepdims=True
        )
        one_hot = np.zeros_like(logits.data)
        one_hot[np.arange(logits.data.shape[0]), targets] = 1.0
        expected_gradient = (probabilities - one_hot) / logits.data.shape[0]

        np.testing.assert_allclose(logits.grad, expected_gradient)


if __name__ == "__main__":
    unittest.main()
