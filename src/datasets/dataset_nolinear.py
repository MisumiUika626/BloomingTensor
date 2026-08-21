class Dataset:
    def __init__(self):

        self.samples = [
            ([-2.0], 4.0),
            ([-1.5], 2.25),
            ([-1.0], 1.0),
            ([-0.5], 0.25),
            ([0.0], 0.0),
            ([0.5], 0.25),
            ([1.0], 1.0),
            ([1.5], 2.25),
            ([2.0], 4.0),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
