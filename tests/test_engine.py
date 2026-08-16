import unittest

from src.autograd.engine import Value


class TestValue(unittest.TestCase):
    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)

        result = a + b
        result.backward()

        self.assertAlmostEqual(result.data, 5.0)
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, 1.0)

    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)

        result = a * b
        result.backward()

        self.assertAlmostEqual(result.data, 6.0)
        self.assertAlmostEqual(a.grad, 3.0)
        self.assertAlmostEqual(b.grad, 2.0)

    def test_shared_node_accumulates_gradient(self):
        x = Value(2.0)

        result = x * x + x
        result.backward()

        self.assertAlmostEqual(result.data, 6.0)
        self.assertAlmostEqual(x.grad, 5.0)

    def test_chain_rule(self):
        a = Value(2.0)
        b = Value(3.0)

        c = a + b
        loss = c * a
        loss.backward()

        self.assertAlmostEqual(loss.data, 10.0)
        self.assertAlmostEqual(c.grad, 2.0)
        self.assertAlmostEqual(a.grad, 7.0)
        self.assertAlmostEqual(b.grad, 2.0)

    def test_power(self):
        x = Value(3.0)

        result = x ** 2
        result.backward()

        self.assertAlmostEqual(result.data, 9.0)
        self.assertAlmostEqual(x.grad, 6.0)

    def test_operations_with_constants(self):
        x = Value(2.0)

        result = 3 * x + 4
        result.backward()

        self.assertAlmostEqual(result.data, 10.0)
        self.assertAlmostEqual(x.grad, 3.0)

    def test_reverse_subtraction(self):
        x = Value(2.0)

        result = 5 - x
        result.backward()

        self.assertAlmostEqual(result.data, 3.0)
        self.assertAlmostEqual(x.grad, -1.0)


if __name__ == "__main__":
    unittest.main()