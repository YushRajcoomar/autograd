from .neuron import Neuron


class Layer:
    def __init__(self, nin, nouts: list) -> None:
        self.neurons = [Neuron(nin) for _ in range(nouts)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
