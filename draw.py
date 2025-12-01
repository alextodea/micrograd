from graphviz import Digraph
from value import Value


class Draw:
    def __init__(self, value: Value) -> None:
        self.value = value

    def trace(self, root: Value):
        # build a set of all nodes and edges in a graph
        nodes, edges = set(), set()

        def build(v: Value):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    def draw_dot(self):
        root = self.value
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

        nodes, edges = self.trace(root)
        for n in nodes:
            uid = str(id(n))
            # for any value in graph, create a rectanglular node for it
            dot.node(
                name=uid,
                label="{%s | data %.4f | grad  %.4f}" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                # if Value has operator then create a node out of it
                dot.node(
                    name=uid + n._op,
                    label=n._op,
                )
                # connect value node with operation node
                dot.edge(
                    uid + n._op,
                    uid,
                )

        for n1, n2 in edges:
            # connect n1 to n2 of operator
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
