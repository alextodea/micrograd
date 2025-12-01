import math


class Value:

    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other_value):
        other_value = (
            other_value if isinstance(other_value, Value) else Value(other_value)
        )
        out = Value(self.data + other_value.data, (self, other_value), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other_value.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other_value):
        return self + (-other_value)

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other_value):  # other * self
        return self * other_value

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __mul__(self, other_value):
        other_value = (
            other_value if isinstance(other_value, Value) else Value(other_value)
        )
        out = Value(self.data * other_value.data, (self, other_value), "*")

        def _backward():
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def __pow__(self, other_value):
        assert isinstance(
            other_value, (int, float)
        ), "only supporting int/float powers for now"

        out = Value(self.data**other_value, (self,), f"**{other_value}")

        def _backward():
            self.grad += other_value * (self.data ** (other_value - 1)) * out.grad

        out._backward = _backward

        return out
