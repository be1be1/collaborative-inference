from graphviz import Digraph
import torch
from torch.autograd import Variable
from dnn_models.GoogLeNet import *
from torchvision import models
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    G = nx.DiGraph()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
                G.add_node(str(id(var)), name=str(type(var).__name__))
                print("just add node %s, the name is %s" % (str(id(var)), str(type(var).__name__)))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        if str(type(u[0]).__name__) != "AccumulateGrad":
                            G.add_edge(str(id(u[0])), str(id(var)))
                            print("add an edge from %s node to %s node" % (str(type(u[0]).__name__), str(type(var).__name__)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    G.add_edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot, G


if __name__ == '__main__':
    inputs = torch.randn(1, 1, 96, 96)
    inception = GoogLeNet()
    y = inception(Variable(inputs))
    dot, G = make_dot(y, inception.state_dict())
    dot.view(filename="1.dot", directory="./")
    # labels = nx.get_node_attributes(G, 'name')
    # nx.draw(G, labels=labels)
    # plt.show()
    # print([sorted(generation) for generation in nx.topological_generations(G)])



