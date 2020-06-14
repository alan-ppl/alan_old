import string
import torch as t
from torch.distributions import Normal


#### Workarounds because torch.distributions doesn't know about named tensors
def unify_name(*names):
    not_none_names = [name for name in names if name is not None]
    assert all(name == not_none_names[0] for name in not_none_names)
    if 0 < len(not_none_names):
        assert isinstance(not_none_names[0], str)
        return not_none_names[0]
    else:
        return None

def unify_names(*nss):
    length = max([0, *[len(ns) for ns in nss]])
    padded_nss = [ [*((length-len(ns))*(None,)), *ns] for ns in nss]
    return [unify_name(*ns) for ns in zip(*padded_nss)]

def unify_arg_names(*args):
    return unify_names(*(arg.names for arg in args if isinstance(arg, t.Tensor)))

def strip_name(arg):
    if isinstance(arg, t.Tensor):
        return arg.rename(None)
    else:
        return arg
def strip_names(*args):
    return (strip_name(arg) for arg in args)

class WrappedDist:
    def __init__(self, dist, *args):
        self.unified_names = unify_arg_names(*args)
        self.dist = dist(*strip_names(*args))
    def rsample(self, sample_shape=t.Size([]), new_names=[]):
        return self.dist.rsample(sample_shape=sample_shape).refine_names(*new_names, *self.unified_names)
    def log_prob(self, x):
        return self.dist.log_prob(x).refine_names(*unify_names(x.names, self.unified_names))


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

def positional_arg_names(dims):
    return [f"pos{i}" for i in string.ascii_uppercase[:dims]]

def plate_dim_names(key, dims):
    if dims==1:
        return [f"plate_{key}"]
    else:
        return [f"plate_{key}{i}" for i in string.ascii_uppercase[:dims]]

class SampleLogProbK():
    """
    Draws K samples from the model, keeping straight line dependencies:
    i.e. z_2^k depends on z_1^k
    names: [*plates, _K, *pos]
    """
    def __init__(self, K, protected_dims):
        self.K = K
        self.protected_dims = protected_dims
        self.plate_names = []

    def __call__(self, prefix_trace, dist, plate, in_kwargs):
        data = in_kwargs["data"]

        plate_names = plate_dim_names(prefix_trace.prefix, len(plate))
        assert all(pn not in self.plate_names for pn in plate_names)
        self.plate_names = plate_names + self.plate_names

        if data is not None:
            assert plate == t.Size([])
            return data, {}
        else:
            new_dim_shape_no_pad = [*plate]
            new_dim_names_no_pad = plate_names

            if "_K" not in dist.unified_names:
                new_dim_shape_no_pad.append(self.K)
                new_dim_names_no_pad.append("_K")
            padding = max(0, self.protected_dims - len(dist.unified_names))

            new_dim_shape = [*new_dim_shape_no_pad, *(padding*[1])]
            new_dim_names = [*new_dim_names_no_pad, *positional_arg_names(padding)]

            sample = dist.rsample(sample_shape=t.Size(new_dim_shape), new_names=new_dim_names)

            # align to global plate names
            sample = sample.refine_names(*self.plate_names, "_K", *positional_arg_names(self.protected_dims))
            lp = dist.log_prob(sample)
            return sample, {"sample": sample, "log_prob": lp}

class LogProbK():
    """
    Evaluates probabilities of fixed samples for all combinations.
    names: [*arg, *plates, *pos]
    """
    def __init__(self, protected_dims):
        self.protected_dims = protected_dims
        self.pos_names = positional_arg_names(protected_dims)
        self.arg_names = []
        self.plate_names = []

    def __call__(self, prefix_trace, dist, plate, in_kwargs):
        sample, data = in_kwargs["sample"], in_kwargs["data"]

        assert (sample is None) or (data is None)
        if data is not None:
            return data, {"log_prob": dist.log_prob(data)}
        else:
            new_dim = prefix_trace.prefix
            self.arg_names.append(new_dim)

            # rename _K, taking account of plates
            sample = sample.rename(*self.plate_names, new_dim, *self.pos_names)
            # align sample to new ordering
            sample = sample.align_to(*self.arg_names[::-1], *self.plate_names, *self.pos_names)
            
            log_prob = dist.log_prob(sample)
            return sample, {"sample": sample, "log_prob": log_prob}
  
class PrefixTrace():
    def __init__(self, prefix, trace):
        self.prefix = prefix
        self.trace = trace

    def __getitem__(self, key):
        return PrefixTrace(self.prefix + "/" + key, self.trace)

    def __call__(self, dist, plate=t.Size([])):
        """
        compute out_dicts from in_dicts for the current primitive
        """
        in_dicts = self.trace.in_dicts
        out_dicts = self.trace.out_dicts
        fn = self.trace.fn

        # use self.prefix to index into all the in_dicts, returning None if the in_dict is empty
        in_kwargs = {}
        for key, in_dict in in_dicts.items():
            in_kwargs[key] = in_dict.get(self.prefix)

        # map in_kwargs to out_kwargs using self.fn
        result, out_kwargs = fn(self, dist, plate, in_kwargs)

        # put out_kwargs into out_dicts, not
        for key in out_kwargs:
            if key not in out_dicts:
                out_dicts[key] = {}
            out_dicts[key][self.prefix] = out_kwargs[key]
        
        return result

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

    def __getitem__(self, key):
        return PrefixTrace(key, self)



def dist(trace):
    a = trace["a"](WrappedDist(Normal, t.ones(3), 3), plate=t.Size([3]))
    b = trace["b"](WrappedDist(Normal, a, 3), plate=t.Size([4]))
    return (a+b)


tr1 = Trace({"data": {}}, SampleLogProbK(4, 2))
val = dist(tr1)
tr2 = Trace({"data": {}, "sample": tr1.out_dicts["sample"]}, LogProbK(2))
val = dist(tr2)
