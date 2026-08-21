import numpy as np


class Dataset:
    def __init__(self, n_samples=200, noise=0.15, random_seed=42):
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if noise < 0:
            raise ValueError("noise must be non-negative")

        first_count = n_samples // 2
        second_count = n_samples - first_count

        first_angles = np.linspace(0.0, np.pi, first_count)
        second_angles = np.linspace(0.0, np.pi, second_count)

        first_moon = np.column_stack((np.cos(first_angles), np.sin(first_angles)))
        second_moon = np.column_stack(
            (1.0 - np.cos(second_angles), 0.5 - np.sin(second_angles))
        )

        features = np.vstack((first_moon, second_moon))
        targets = np.concatenate(
            (np.zeros(first_count, dtype=int), np.ones(second_count, dtype=int))
        )

        random = np.random.default_rng(random_seed)
        features += random.normal(0.0, noise, size=features.shape)

        order = random.permutation(n_samples)
        features = features[order]
        targets = targets[order]

        self.samples = [
            (feature.tolist(), int(target))
            for feature, target in zip(features, targets)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
