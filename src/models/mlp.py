from ..nn.activations import LeakyReLU
from ..nn.linear import Linear
from ..nn.sequential import Sequential


class MLP:
    def __init__(self, layer_sizes: list[int]):
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            linear = Linear(in_features, out_features)
            layers.append(linear)
            if i != len(layer_sizes) - 2:
                layers.append(LeakyReLU())

        self.network = Sequential(layers)

    def forward(self, x):
        return self.network.forward(x)
