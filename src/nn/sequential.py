class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        all_parameters = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                all_parameters.extend(layer.parameters())
        return all_parameters
