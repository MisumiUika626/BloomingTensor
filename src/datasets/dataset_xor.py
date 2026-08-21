class Dataset:
    def __init__(self):

        self.samples = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
