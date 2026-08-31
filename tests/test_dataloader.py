import unittest

import numpy as np

from src.datasets.dataloader import DataLoader


class SmallDataset:
    def __init__(self):
        self.images = np.arange(15).reshape(5, 3)
        self.labels = np.arange(5)

    def __len__(self):
        return len(self.labels)


class TestDataLoader(unittest.TestCase):
    def test_batches_without_shuffle_include_partial_last_batch(self):
        loader = DataLoader(SmallDataset(), batch_size=2, shuffle=False)

        batches = list(loader)

        self.assertEqual(len(loader), 3)
        self.assertEqual(
            [images.shape for images, _ in batches],
            [(2, 3), (2, 3), (1, 3)],
        )
        np.testing.assert_array_equal(batches[0][1], [0, 1])
        np.testing.assert_array_equal(batches[-1][1], [4])

    def test_same_seed_produces_same_first_epoch_order(self):
        first_loader = DataLoader(
            SmallDataset(), batch_size=2, shuffle=True, random_seed=7
        )
        second_loader = DataLoader(
            SmallDataset(), batch_size=2, shuffle=True, random_seed=7
        )

        first_order = np.concatenate([labels for _, labels in first_loader])
        second_order = np.concatenate([labels for _, labels in second_loader])

        np.testing.assert_array_equal(first_order, second_order)
        self.assertEqual(set(first_order.tolist()), set(range(5)))

    def test_non_positive_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            DataLoader(SmallDataset(), batch_size=0)


if __name__ == "__main__":
    unittest.main()
