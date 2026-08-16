class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x
