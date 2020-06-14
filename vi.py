import named_tensor
import trace

def P(trace):
    a = normal(trace["a"], 2, 3)
    b = normal(trace["b"], 2, 3)
    return (a+b)

class Q(nn.Module)
    def __init__(self)
