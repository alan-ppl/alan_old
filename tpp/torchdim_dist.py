import torch as t
import torch.distributions as td
from functorch.dim import dims, Tensor

def univariate(*names):
    return ({name: 0 for name in names}, 0)
univariate_loc_scale = univariate("loc", "scale")

param_event_ndim = {
    "Bernoulli":                 univariate("probs", "logits"),
    "Beta":                      univariate("concentration1", "concentration0"),
    "Binomial":                  univariate("total_count", "probs", "logits"),
    "Categorical":               ({"probs": 1, "logits": 1}, 0),
    "Cauchy":                    univariate_loc_scale,
    "Chi2":                      univariate("df"),
    "ContinuousBernoulli":       univariate("probs", "logits"),
    "Exponential":               univariate("rate"),
    "FisherSnedecor":            univariate("df1", "df2"),
    "Gamma":                     univariate("concentration", "rate"),
    "Geometric":                 univariate("probs", "logits"),
    "Gumbel":                    univariate_loc_scale,
    "HalfCauchy":                univariate("scale"),
    "HalfNormal":                univariate("scale"),
    "Kumaraswamy":               univariate("concentration1", "concentration0"),
    "LKJCholesky":               ({"dim":0, "concentration":0}, 2),
    "Laplace":                   univariate_loc_scale,
    "LogNormal":                 univariate_loc_scale,
    "LowRankMultivariateNormal": ({"loc":1, "cov_factor":2, "cov_diag": 1}, 1),
    "Multinomial":               ({"total_count": 0, "probs": 1, "logits": 1}, 1),
    "MultivariateNormal":        ({"loc": 1, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 1}),
    "NegativeBinomial":          univariate("total_count", "probs", "logits"),
    "Normal":                    univariate_loc_scale,
    "Pareto":                    univariate("scale", "alpha"),
    "Poisson":                   univariate("rate"),
    "RelaxedBernoulli":          univariate("temperature", "probs", "logits"),
    "LogitRelaxedBernoulli":     univariate("temperature", "probs", "logits"),
    "RelaxedOneHotCategorical":  ({"temperature": 0, "probs": 1, "logits": 1}, 0),
    "StudentT":                  univariate("df", "loc", "scale"),
    "Uniform":                   univariate("low", "high"),
    "VonMises":                  univariate("loc", "concentration"),
    "Weibull":                   univariate("scale", "concentration"),
    "Wishart":                   ({"df": 0, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 2}, 2)
}

def pad_nones(arg, ndim):
    """
    Pad with as many unnamed dimensions as necessary to reach ndim
    Ignores torchdim dimensions
    """
    if isinstance(arg, (t.Tensor, Tensor)):
        idxs = (ndim - arg.ndim)*[None]
        idxs.append(Ellipsis)
        return arg.__getitem__(idxs)
    else:
        return arg

def generic_ndim(x):
    assert isinstance(x, (t.Tensor, Tensor, int, float))
    return x.ndim if isinstance(x, (t.Tensor, Tensor)) else 0

def generic_dims(x):
    return x.dims if isinstance(x, Tensor) else ()

def generic_order(x, dims):
    return x.order(dims) if isinstance(x, Tensor) else x
    

class TorchDimDist():
    def __init__(self, dist_name, *args, **kwargs):
        self.dist_name = dist_name
        param_ndim, self.result_ndim = param_event_ndim[dist_name]
        self.dist = getattr(td, dist_name)

        #convert all args to kwargs, assuming that the arguments in param_event_ndim have the right order
        arg_dict = {argname: args[i] for (i, argname) in enumerate(param_ndim.keys())}
        #Merge args and kwargs into a unified kwarg dict
        self.torchdim_args = {**arg_dict, **kwargs}
        #No overlaps in the names for arg_dict and kwargs
        assert len(self.torchdim_args) == len(kwargs) + len(arg_dict)

        #There may be unnamed dimensions over which we're batching.
        unnamed_batch_dim_list = tuple(generic_ndim(arg) - param_ndim[argname] for (argname, arg) in self.torchdim_args.items())
        assert all(0<=x for x in unnamed_batch_dim_list)
        self.unnamed_batch_dims = max(unnamed_batch_dim_list)

        #Save all torchdims to compare against sample_shape
        self.all_torchdims = sorted(set([item for arg in self.torchdim_args.values() for item in generic_dims(arg)]))

        self.tensor_args = {}
        for (argname, arg) in self.torchdim_args.items():
            #Pad all args up to the right lengths.
            arg = pad_nones(arg, self.unnamed_batch_dims+param_ndim[argname])
            #And remove the torchdims
            arg = generic_order(arg, self.all_torchdims)
            self.tensor_args[argname] = arg

    def generic_sample(self, f, sample_dims):
        sample_method = getattr(self.dist(**self.tensor_args), f)
        sample_dims = sorted(set(sample_dims).difference(self.all_torchdims))
        sample_shape = [dim.size for dim in sample_dims]
        sample = sample_method(sample_shape=sample_shape)
        print(sample.shape)
        return sample.__getitem__([*sample_dims, *self.all_torchdims, Ellipsis])

    def rsample(self, sample_dims=()):
        return self.generic_sample("rsample", sample_dims)

    def sample(self, sample_dims=()):
        return self.generic_sample("sample", sample_dims)

    def log_prob(self, x):
        return self.dist(**self.tensor_args)
        
def set_dist(dist_name):
    def inner(*args, **kwargs):
        return TorchDimDist(dist_name, *args, **kwargs)
    globals()[dist_name] = inner

for dist_name in param_event_ndim:
    set_dist(dist_name)

i = dims(1)
j = dims(1, [10])
mean = t.ones(3,3)[i]
std = t.ones(())
dist = Normal(mean, std)
result = dist.sample(sample_dims=(j,))
