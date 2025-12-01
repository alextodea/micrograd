from nn import MLP

from draw import Draw
from value import Value


x = [2.0, 4.0, -1.0]
mlp = MLP(3, [3, 2, 1])

# Number of layers in MLP
print(f"Total number of layers in the MLP: {len(mlp.layers)}\n")

# Iterate through each layer and inspect its details
for idx, layer in enumerate(mlp.layers, 1):
    print(f"Layer {idx}:")

    # Number of neurons in the current layer
    print(f"Number of Neurons: {len(layer.neurons)}")

    # Inspect each neuron's weights and bias
    for neuron_idx, neuron in enumerate(layer.neurons, 1):
        weights = [w.data for w in neuron.w]
        bias = neuron.b.data
        print(f"    Neuron {neuron_idx} weights: {weights}")
        print(f"    Neuron {neuron_idx} bias: {bias}")

    print("")

params = mlp.parameters()
print(f"Parameters in the MLP: {params} \n")

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 5.1], [5.2, 1.1, 2.0]]
ys = [-1.0, 1.0, 1.0]

for k in range(20):
    ypred = [mlp(x) for x in xs]
    ypred = [v[0] for v in ypred]

    # forward pass
    loss: Value = Value(0)
    for ygt, yout in zip(ys, ypred):
        print(f"Desired y: {ygt} and predicted y: {round(yout.data, 3)} \n")
        this_loss: Value = (yout - ygt) ** 2
        assert isinstance(this_loss, Value)
        loss += this_loss

    # backward pass
    for p in mlp.parameters():
        p.grad = 0.0

    loss.grad = 1.00
    loss.backward()

    # update
    for p in mlp.parameters():
        p.data += -0.2 * p.grad

    print(k, loss.data)
