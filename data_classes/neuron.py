from .value import Value

import numpy as np


class Neuron:
    def __init__(self, nin) -> None:
        self.nin = nin
        self.w = [
            Value(np.random.uniform(-1, 1), label=f"w_{i}")
            for i, _ in enumerate(range(self.nin))
        ]
        self.b = Value(np.random.uniform(-1, 1), label="b")

    def __call__(self, x):
        # tanh(w*x +b)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.softmax()
        return out

    def parameters(self):
        return self.w + [self.b]
