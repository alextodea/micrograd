import random
from value import Value


class Neuron:

    def __init__(self, nr_inputs) -> None:
        self.w: list[Value] = [Value(random.uniform(-1, 1)) for _ in range(nr_inputs)]
        self.b: Value = Value(random.uniform(-1, 1))

    def __call__(self, x: list) -> Value:
        sum_xw_b: Value = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        assert isinstance(sum_xw_b, Value)
        return sum_xw_b.tanh()

    def parameters(self) -> list:
        return self.w + [self.b]


class Layer:
    def __init__(self, neurons_in, nr_outputs) -> None:
        self.neurons = [Neuron(nr_inputs=neurons_in) for _ in range(nr_outputs)]

    def __call__(self, x) -> list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self) -> list:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, input_size: int, output_sizes: list[int]) -> None:
        sizes = [input_size] + output_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(output_sizes))]

    def __call__(self, x) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]
