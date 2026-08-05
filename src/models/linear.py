import numpy as np

from ..autograd.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / np.sqrt(in_features)
        # 对于tensor_w,axis=0表示输入的特征数，axis=1表示输出神经元数（特征数）
        weight_data = np.random.uniform(
            -bound,
            bound,
            size=(in_features, out_features),
        )
        bias_data = np.zeros(out_features)
        self.weight = Tensor(weight_data)
        self.bias = Tensor(bias_data)

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor")
        if x.data.ndim != 2:
            raise ValueError(
                f"Input must be a 2D Tensor with shape (batch_size, {self.in_features})"
            )
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Input features must match in_features: expected {self.in_features}, got {x.data.shape[1]}"
            )
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
