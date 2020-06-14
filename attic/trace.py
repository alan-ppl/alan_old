import torch
from named_tensors import broadcast_args, NTensor, ndistributions
normal = ndistributions.normal
gamma = ndistributions.gamma


def sample_log_prob(dist, data):
    """
    initializes with a random sample from the prior
    {"data"} => {"sample", "log_prob"}
    """
    if data is not None:
        log_prob = dist.log_prob(data)
        return data, {}
    else:
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        return sample, {"sample": sample, "log_prob": log_prob}

def log_prob(dist, data, sample):
    """
    initializes with a random sample from the prior
    {"data", sample} => {"log_prob"}
    """
    assert (data is None) != (sample is None)
    result = data if (data is not None) else sample
    return result, {"log_prob": dist.log_prob(result)}
  

class Trace:
    """
    in_trace and out_trace are dicts
    calling trace["a"] gives back a new object, with refs to the same dicts, with an extended prefix

    in_dicts and out_dicts are dictionaries
    
    different standard dicts include:
      data (value of observed data)
      sample (value of latent variables)
      log_prob (log-probability of data and latent variables)
    """
    def __init__(self, in_dicts, fn):
        assert "data" in in_dicts
        self.in_dicts = in_dicts
        self.out_dicts = {}
        self.fn = fn
        self.prefix = ""

    def __getitem__(self, key):
        # return a trace with refs to all the same underlying objects, but with an updated prefix
        trace = Trace(self.in_dicts, self.fn)
        trace.prefix = self.prefix + "/" + key
        trace.out_dicts = self.out_dicts
        return trace

    def primitive(self, dist, K=None, group=None):
        """
        compute out_dicts from in_dicts for the current primitive
        """
        assert K is None
        assert group is None

        # use self.prefix to index into all the in_dicts, returning None if the in_dict is empty
        in_kwargs = {}
        for key, in_dict in self.in_dicts.items():
            in_kwargs[key] = in_dict.get(self.prefix)

        # map in_kwargs to out_kwargs using self.fn
        result, out_kwargs = self.fn(dist, **in_kwargs)

        # put out_kwargs into out_dicts, not
        for key in out_kwargs:
            if key not in self.out_dicts:
                self.out_dicts[key] = {}
            self.out_dicts[key][self.prefix] = out_kwargs[key]
        
        return result



def dist(trace):
    a = normal(trace["a"], 2, 3)
    b = normal(trace["b"], 2, 3)
    return (a+b)

t = Trace({"data": {}}, sample_log_prob)
val = dist(t)
