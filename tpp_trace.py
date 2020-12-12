"""
    Helpers to manage not just Pytorch named dimensions, 
    but three types of named dimension with their own orderings:
    plate dims, sample dim, and underlying latent dims.
    Also trace objects to track the tensors at each point.
"""
import re
import string
import torch as t
import torch.distributions as td


def unify_names(*nss):
    result = sum(t.zeros(len(ns)*(0,), names=ns) for ns in nss)
    return result.names

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
        return self.dist.rsample(sample_shape=sample_shape) \
                .refine_names(*new_names, *self.unified_names)
    
    def log_prob(self, x):
        return self.dist.log_prob(x) \
                .refine_names(*unify_names(x.names, self.unified_names))



def concat_prefix(prefix, key):
    return prefix + "__" + key


def pos_name(i) :
    return f"pos_{i}"


def positional_dim_names(dims):
    return [pos_name(i) for i in string.ascii_uppercase[:dims]]


def plate_dim_names(key, dims):
    if dims==1:
        return [f"plate_{key}"]
    else:
        return [f"plate_{key}_{i}" for i in string.ascii_uppercase[:dims]]

    
def k_dim_name(addr):
    return f"_k{addr}"


class SampleLogProbK():
    """
    Draws K samples from the model, keeping straight line dependencies:
    i.e. z_2^k depends on z_1^k
    names: [*plates, _K, *pos]
    """
    def __init__(self, K, protected_dims):
        self.dim_names = ["_K", *(protected_dims*(None,))]
        self.dim_shape = [K, *(protected_dims*(1,))]
    
    def __call__(self, prefix_trace, dist, plate_name, plate_shape, in_kwargs):
        assert "data" not in in_kwargs
        
        if plate_name is not None:
            plate_name = "_plate_" + plate_name
            assert plate_name not in self.dim_names
            self.dim_names.insert(0, plate_name)
            self.dim_shape.insert(0, 1)
        
        if plate_name is None:
            dim_shape = self.dim_shape
        else:
            dim_shape = [plate_shape, *self.dim_shape[1:]]

        #how much of the final shape is missing in dist.unified_names
        N = len(self.dim_names) - len(dist.unified_names) 

        #the shape and dims we need to add to make it up to the right shape.
        sample_shape = dim_shape[:N]
        sample_names = self.dim_names[:N]
        
        sample = dist.rsample(sample_shape=t.Size(sample_shape), new_names=sample_names)
        lps = dist.log_prob(sample)
        
        return sample, {"sample": sample, "log_prob": lps}

    def delete_names(self, trace_prefix, names, tensors):
        return tensors


def squeeze_dims(tensor, dims):
    for dim in dims:
        tensor = tensor.squeeze(dim)
    
    return tensor


class LogProbK():
    """
    Evaluates probabilities of fixed samples for all combinations.
    Takes plate_names as input
    names: [*arg, *plates, *pos]
    """
    def __init__(self, dim_names): #plate_names, protected_dims):
        self.protected_dims = sum(x==None for x in dim_names)
        self.plate_names = dim_names[:-(protected_dims+1)]

        self.pos_names = positional_dim_names(self.protected_dims)

        self.arg_names = []
        self.ordered_plate_names = []

    def __call__(self, prefix_trace, dist, plate_name, plate_shape, in_kwargs):
        sample, data = in_kwargs["sample"], in_kwargs["data"]

        assert (sample is None) or (data is None)


        if data is not None:
            assert plate_name is None
            return data, {"log_prob": dist.log_prob(data)}
        else:

            if plate_name is not None:
                plate_name = "_plate_" + plate_name
                assert plate_name not in self.ordered_plate_names
                self.ordered_plate_names.append(plate_name)
                assert plate_name in sample.names

            new_dim = k_dim_name(prefix_trace.prefix)
            self.arg_names.append(new_dim)

            # name positional dims
            sample = sample.refine_names(..., *self.pos_names)
            # introduce dims for all plates,
            sample = sample.align_to(*self.plate_names, "_K", *self.pos_names)
            # rename _K, taking account of plates
            sample = sample.rename(*self.plate_names, new_dim, *self.pos_names)
            # align sample to new ordering
            sample = sample.align_to(*self.arg_names[::-1], \
                                     *self.plate_names, \
                                     *self.pos_names)
            # remove positional names
            sample = sample.rename(*sample.names[:-self.protected_dims], *(self.protected_dims*(None,)))
            
            log_prob = dist.log_prob(sample)
            return sample, {"sample": sample, "log_prob": log_prob}

    def delete_names(self, trace_prefix, names, tensors):
        """
        Avoids the problem of too many dimensions.
        Removes arg_names, and squeezes corresponding dimensions out of tensors.
        Must put ALL tensors that you're going to use in future through this function, 
        otherwise dims won't align
        """
        names = [k_dim_name(concat_prefix(trace_prefix.prefix, name)) for name in names]

        for name in names:
            # `remove` intentionally causes error if name isn't present.
            self.arg_names.remove(name)

        # squeeze errors if name isn't present, or if dimension shape is more than one
        return [squeeze_dims(tensor, names) for tensor in tensors]


# model -> list of sample or log_prob tensors
tclass PrefixTrace():
    def __init__(self, prefix, trace):
        self.prefix = prefix
        self.trace = trace

    def __getitem__(self, key):
        return PrefixTrace(concat_prefix(self.prefix, key), self.trace)


    def call(self, dist, plate_name=None, plate_shape=None):
        """
        compute out_dicts from in_dicts for the current primitive
        """
        in_dicts = self.trace.in_dicts
        out_dicts = self.trace.out_dicts
        fn = self.trace.fn

        # use self.prefix to index into all the in_dicts, 
        # returning None if the in_dict is empty
        in_kwargs = {}
        for key, in_dict in in_dicts.items():
            in_kwargs[key] = in_dict.get(self.prefix)

        # map in_kwargs to out_kwargs using self.fn
        result, out_kwargs = fn(self, dist, plate_name, plate_shape, in_kwargs)

        # put out_kwargs into out_dicts, not
        for key in out_kwargs:
            if key not in out_dicts:
                out_dicts[key] = {}
            out_dicts[key][self.prefix] = out_kwargs[key]
        
        return result

    def delete_names(self, *args, **kwargs):
        return self.trace.fn.delete_names(self, *args, **kwargs)
    
    # Traverse out_dicts and sum out positional dims inplace
    def sum_out_pos(self) :
        for kind, d in self.trace.out_dicts.items() :
            for k, T in d.items() :
                posses = [dim for dim in T.names \
                          if pos_name("") in dim]
                for pos in posses :
                    d[k] = d[k].sum(pos)

dist_names = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Exponential",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Laplace",
    "LogNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "RelaxedBernoulli",
    "LogitRelaxedBernoulli",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull",
]

#Multivariate!
#Dirichlet
#MvNormal
#Multinomial
#OneHotCategorical
#RelaxedOneHotCategorical

for dist_name in dist_names:
    def f(self, *args, plate_name=None, plate_shape=None, **kwargs):
        dist = WrappedDist(getattr(td, dist_name), *args, **kwargs)
        return self.call(dist, plate_name=plate_name, plate_shape=plate_shape)
    setattr(PrefixTrace, dist_name, f)


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
        self.in_dicts = in_dicts
        self.out_dicts = {}
        self.fn = fn

    def __getitem__(self, key):
        return PrefixTrace(key, self)


def trace(in_dicts, fn):
    return PrefixTrace("", Trace(in_dicts, fn))


def rename_placeholders(reference, placeheld) :
    """
    Rename and align in Q trace
    The ks are misaligned between trp and trq; Q's trace only has "__K"
    a placeholder for the current sampled var
    """
    reference_dict = reference.trace.out_dicts
    placeheld_dict = placeheld.trace.out_dicts
    
    # for sample dict and log_prob dict
    for k, d in placeheld_dict.items() :
        for var, tensor in d.items() :
            tensor = tensor.rename(_K = k_dim_name(var))
            target = reference_dict[k][var]
            placeheld_dict[k][var] = tensor.align_as(target)
            
    return placeheld_dict


def subtract_latent_log_probs(trp, trq) :
    p_dict = trp.trace.out_dicts["log_prob"]
    q_dict = trq.trace.out_dicts["log_prob"]
    
    # Check there are only latents left   
    # P also has log_probs from the observed vars. 
    # so looping over Q should do it
    tensors = {}
    for k, Q in q_dict.items() :
        tensors[k] = p_dict[k] - Q
    
    return tensors


def sampler(draws, nProtected):
    return trace({}, SampleLogProbK(K=draws, protected_dims=nProtected))


def evaluator(wrapper, nProtected, data={}) :
    tr = wrapper.trace
    samples = tr.out_dicts.get("sample")
    assert(samples is not None)
    
    d = {"data": data, "sample": samples} 
    lpk = LogProbK(plate_names=tr.fn.plate_names, protected_dims=nProtected)
    
    return trace(d, lpk)


def sample_and_eval(model, draws, nProtected, data={}, verbose=False) :
    tr1 = sampler(draws, nProtected, data=data)
    val = model(tr1)
    
    sampled_names = val.names
    
    tr2 = evaluator(tr1, nProtected, data=data)
    val = model(tr2)
    
    if verbose :
        print("Plates", tr1.trace.fn.plate_names)
        print(sampled_names)
        print(val.names)
        print()
    
    return tr2


# example directed graph without plate repeats
# the real stuff happens as side effects to trace's dicts
def chain_dist(trace):
    a = trace["a"](WrappedDist(Normal, t.ones(3), 3))
    b = trace["b"](WrappedDist(Normal, a, 3))
    c = trace["c"](WrappedDist(Normal, b, 3))
    (c,) = trace.delete_names(("a", "b"), (c,))
    d = trace["d"](WrappedDist(Normal, c, 3))
    
    return d


# example directed graph with plate repeats
# 3(a) -> 4(b) -> c -> d
def plate_dist(trace):
    a = trace["a"].Normal(t.ones(3), 3, plate_name="A", plate_shape=3)
    b = trace["b"].Normal(a, 3,         plate_name="B", plate_shape=4)
    c = trace["c"].Normal(b, 3)
    (c,) = trace.delete_names(("a", "b"), (c,))
    d = trace["d"].Normal(c, 3)
    
    return d


# TODO: Work out odd munmap_chunk(): invalid pointer bug onquit
if __name__ == "__main__" :
    protected_dims = 2
    tr1 = trace({}, SampleLogProbK(K=4, protected_dims=protected_dims))
    val = plate_dist(tr1)
    tr2 = trace({"data": {}, "sample": tr1.trace.out_dicts["sample"]}, LogProbK(tr1.trace.fn.dim_names))
    val = plate_dist(tr2)
    
    #print(tr2.trace.out_dicts["log_prob"])

        

