import unittest

import numpy as np

from src.autograd.tensor import Tensor


class TestTensorAutograd(unittest.TestCase):
    def test_add_same_tensor_accumulates_gradient(self):
        x = Tensor([1.0, 2.0, 3.0])
        loss = (x + x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, [2.0, 2.0, 2.0])

    def test_multiply_same_tensor_gradient_is_two_x(self):
        x = Tensor([-2.0, 0.0, 3.0])
        loss = (x * x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, 2.0 * x.data)

    def test_gradient_adds_when_variable_has_two_paths(self):
        x = Tensor([1.0, 2.0, 3.0])
        loss = (2.0 * x + x**2).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, 2.0 + 2.0 * x.data)

    def test_broadcast_gradient_returns_to_original_shape(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = Tensor([10.0, 20.0, 30.0])
        loss = (x + bias).sum()
        loss.backward()

        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(bias.grad.shape, bias.data.shape)
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))
        np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])

    def test_matrix_multiplication_gradient_shape_and_value(self):
        left = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        loss = (left @ right).sum()
        loss.backward()

        self.assertEqual(left.grad.shape, (2, 3))
        self.assertEqual(right.grad.shape, (3, 2))
        np.testing.assert_allclose(
            left.grad,
            [[15.0, 19.0, 23.0], [15.0, 19.0, 23.0]],
        )
        np.testing.assert_allclose(
            right.grad,
            [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]],
        )

    def test_relu_gradient_for_positive_negative_and_zero(self):
        x = Tensor([-2.0, 0.0, 3.0])
        loss = x.relu().sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])

    def test_mean_gradient_divides_by_element_count(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = x.mean()
        result.backward()

        np.testing.assert_allclose(result.data, 2.5)
        np.testing.assert_allclose(x.grad, np.full((2, 2), 0.25))

    def test_backward_rejects_non_scalar_tensor(self):
        x = Tensor([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(RuntimeError, "backward.*scalar"):
            x.backward()

    def test_repeated_backward_produces_same_gradient(self):
        x = Tensor([1.0, 2.0, 3.0])
        loss = (x * x).sum()

        loss.backward()
        first_grad = x.grad.copy()
        loss.backward()

        np.testing.assert_allclose(first_grad, [2.0, 4.0, 6.0])
        np.testing.assert_allclose(x.grad, first_grad)

    def test_composite_arithmetic_gradient(self):
        x = Tensor(2.0)
        loss = (x**2 - 2.0 / x) / 2.0
        loss.backward()

        np.testing.assert_allclose(x.grad, 2.25)

    def test_multiplication_broadcast_gradient(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale = Tensor([10.0, 20.0, 30.0])

        loss = (x * scale).sum()
        loss.backward()

        np.testing.assert_allclose(
            x.grad,
            [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]],
        )
        np.testing.assert_allclose(scale.grad, [5.0, 7.0, 9.0])

    def test_exp(self):
        x = Tensor([0, 1, -1])
        out = x.exp()
        loss = (3 * out).sum()
        loss.backward()
        np.testing.assert_allclose(out.data, np.exp([0, 1, -1]))
        np.testing.assert_allclose(x.grad, 3 * np.exp([0, 1, -1]))

    def test_log_forward_and_backward(self):
        x = Tensor([1.0, np.e, np.e**2])
        out = x.log()
        loss = (4 * out).sum()

        loss.backward()

        np.testing.assert_allclose(out.data, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(x.grad, 4 / x.data)

    def test_log_rejects_non_positive_values(self):
        for values in ([1.0, 0.0], [1.0, -1.0]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "positive"):
                    Tensor(values).log()

    def test_leaky_relu(self):
        x = Tensor([-2.0, 0.0, 3.0])
        alpha = 0.01
        out = x.leaky_relu(alpha)
        loss = out.sum()
        loss.backward()
        np.testing.assert_allclose(out.data, [-0.02, 0.0, 3.0])
        np.testing.assert_allclose(x.grad, [alpha, alpha, 1.0])

    def test_invalid_alpha_leaky_relu(self):
        x = Tensor([-2.0, 0.0, 3.0])

        with self.assertRaises(ValueError):
            x.leaky_relu(-0.01)

        with self.assertRaises(ValueError):
            x.leaky_relu(0.0)

        with self.assertRaises(ValueError):
            x.leaky_relu(1.0)

        with self.assertRaises(TypeError):
            x.leaky_relu(True)

        with self.assertRaises(ValueError):
            x.leaky_relu(1.5)


if __name__ == "__main__":
    unittest.main()
