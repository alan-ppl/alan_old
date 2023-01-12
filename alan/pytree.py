import torch
import functorch.dim
Tensor = (torch.Tensor, functorch.dim.Tensor)
Leaf = (str, *Tensor)
from collections.abc import Iterable

def isleaf(xs):
    return isinstance(xs, Leaf) or not isinstance(xs, Iterable)

def values(xs):
    if isleaf(xs):
        yield xs
    else:
        if isinstance(xs, dict):
            xs = xs.values()
        for x in xs:
            yield from values(x)


def treemap(f, tree):
    if isleaf(tree):
        return f(tree)

    if isinstance(tree, dict):
        return {k: treemap(f, subtree) for (k, subtree) in tree.items()}

    result = tuple(treemap(f, subtree) for subtree in tree)

    if   isinstance(tree, (list, set)):
        return type(tree)(result)
    else:
        return result
