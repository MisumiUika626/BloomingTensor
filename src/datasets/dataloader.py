import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, random_seed=42):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = np.random.default_rng(random_seed)

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            images = self.dataset.images[batch_indices]
            labels = self.dataset.labels[batch_indices]
            yield images, labels

    def __len__(self):
        sample_count = len(self.dataset)
        return (sample_count + self.batch_size - 1) // self.batch_size
