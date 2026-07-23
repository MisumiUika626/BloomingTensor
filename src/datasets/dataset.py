class Dataset:

    def __init__(self):

        self.samples = [
            ([1,2,3], 10),
            ([2,3,4], 20),
            ([3,4,5], 30)
        ]


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        return self.samples[index]