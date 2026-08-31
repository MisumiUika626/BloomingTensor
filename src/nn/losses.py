import numpy as np

from ..autograd.tensor import Tensor


class CrossEntropyLoss:
    def forward(self, logits, targets):
        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a Tensor")

        targets = np.asarray(targets, dtype=np.int64)
        row_max_data = np.max(logits.data, axis=1, keepdims=True)
        stable_logits = logits - Tensor(row_max_data)
        # Subtracting each row maximum keeps the exponentials finite.
        log_denominator = stable_logits.exp().sum(axis=1, keepdims=True).log()
        log_probs = stable_logits - log_denominator
        one_hot_data = np.zeros_like(logits.data)
        one_hot_data[np.arange(logits.data.shape[0]), targets] = 1.0
        one_hot = Tensor(one_hot_data)
        selected_log_probs = log_probs * one_hot
        correct_log_probs = selected_log_probs.sum(axis=1)
        loss = (-correct_log_probs).mean()
        return loss
