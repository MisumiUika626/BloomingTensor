import unittest

from src.models.linear import Linear


class TestLinear(unittest.TestCase):
    def test_forward(self):
        model = Linear(input_dim=3)

        output = model.forward([1, 2, 3])

        self.assertAlmostEqual(output, 3.1)

    def test_dimension_mismatch(self):
        model = Linear(input_dim=3)

        with self.assertRaises(ValueError):
            model.forward([1, 2])


if __name__ == "__main__":
    unittest.main()
