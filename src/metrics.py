import numpy as np

from .autograd.tensor import Tensor


def accuracy(logits, targets):
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.argmax(logits.data, axis=1)
    correct = predictions == targets
    correct_count = np.sum(correct)
    total_count = len(correct)
    accuracy_value = correct_count / total_count
    return float(accuracy_value)
