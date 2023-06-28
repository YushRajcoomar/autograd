import numpy as np


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __iter__(self):
        return iter(self)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):  # other + self
        out = Value(self.data + other.data, (self, other), "sum_r")
        return out

    def log(self):
        x = self.data
        out = Value(np.log(x), (self,), "log")
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), _children=(self,), label="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        x = self.data
        t = np.exp(x) / sum(np.exp(m) for m in x)
        out = Value(t, (self,), "softmax")

        def _backward():
            gradient_whole = 0
            for i in range(len(x)):
                for j in range(len(t)):
                    if i == j:
                        gradient_whole += t[i] * (1 - t[j])
                    else:
                        gradient_whole -= t[i] * t[j]
            self.grad += gradient_whole

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()
