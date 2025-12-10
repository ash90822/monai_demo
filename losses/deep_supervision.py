# monai_demo/losses/deep_supervision.py

import torch

class DeepSupervisionLossWrapper:
    def __init__(self, base_loss, weights=None):
        self.base_loss = base_loss
        self.weights = weights

    def __call__(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)

        num_outputs = len(outputs)

        if self.weights is None:
            self.weights = [0.5 ** i for i in range(num_outputs)]

        wsum = sum(self.weights)
        norm_w = [w / wsum for w in self.weights]

        total_loss = 0.0
        for out, w in zip(outputs, norm_w):
            total_loss += w * self.base_loss(out, target)

        return total_loss
