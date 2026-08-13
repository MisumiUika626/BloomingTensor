import numpy as np

from ..autograd.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / np.sqrt(in_features)
        weight_data = np.random.uniform(-bound, bound, size=(in_features, out_features))
        # 此处双层括号：第一层(out_features,)中的,只是参数列表中允许的尾随逗号，传进去仍是整数，
        # 而shape=(out_features,)才是一维turple，因此为了突出shape意图所以用两层括号
        bias_data = np.zeros((out_features,))
        self.weight = Tensor(weight_data)
        self.bias = Tensor(bias_data)

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Type error")
        if x.data.ndim != 2:
            raise ValueError("Need 2d x!")
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, but got {x.data.shape[1]}"
            )
        return x @ self.weight + self.bias

    # 让优化器找到linear的可训练参数
    def parameters(self) -> list[Tensor]:
        # -> list[Tensor]是类型提示，表示“预计返回一个由 Tensor 组成的列表”，但 Python 不会在运行时自动检查它。
        return [self.weight, self.bias]
