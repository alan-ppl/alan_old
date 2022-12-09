import torch as t
import torch.distributions as td
from functorch.dim import dims, Tensor
def univariate(*names):
    return ({name: 0 for name in names}, 0)
univariate_loc_scale = univariate("loc", "scale")

param_event_ndim = {
    "Bernoulli":                 univariate("probs", "logits"),#
    "Beta":                      univariate("concentration1", "concentration0"),#
    "Binomial":                  univariate("total_count", "probs", "logits"),#
    "Categorical":               ({"probs": 1, "logits": 1}, 0),#
    "Cauchy":                    univariate_loc_scale,#
    "Chi2":                      univariate("df"),
    "ContinuousBernoulli":       univariate("probs", "logits"),#
    "Exponential":               univariate("rate"),#
    "FisherSnedecor":            univariate("df1", "df2"),
    "Gamma":                     univariate("concentration", "rate"),
    "Geometric":                 univariate("probs", "logits"),#
    "Gumbel":                    univariate_loc_scale,#
    "HalfCauchy":                univariate("scale"),
    "HalfNormal":                univariate("scale"),
    "Kumaraswamy":               univariate("concentration1", "concentration0"),#
    "LKJCholesky":               ({"dim":0, "concentration":0}, 2),
    "Laplace":                   univariate_loc_scale,#
    "LogNormal":                 univariate_loc_scale,#
    "LowRankMultivariateNormal": ({"loc":1, "cov_factor":2, "cov_diag": 1}, 1),
    "Multinomial":               ({"total_count": 0, "probs": 1, "logits": 1}, 1),#
    "MultivariateNormal":        ({"loc": 1, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 1}, 1),
    "NegativeBinomial":          univariate("total_count", "probs", "logits"),#
    "Normal":                    univariate_loc_scale,#
    "Pareto":                    univariate("scale", "alpha"),
    "Poisson":                   univariate("rate"),#
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
    return x.order(*dims) if isinstance(x, Tensor) else x

def generic_named2dim(x, dims):
    return x if 0==len(dims) else x[dims]

class Dim2Name():
    def __init__(self, names=None, dim2name=None):
        if names is None:
            names = []
        self.names = names

        if dim2name is None:
            dim2name = {}
        self.dim2name = dim2name

    def insert_dim(self, dim):
        if dim in self:
            return self
        else:
            name = repr(dim)
            while name in self.names:
                name = name + "_"
            self = Dim2Name([name, *self.names], {dim: name, **self.dim2name})
        return self

    def insert_tensors(self, tensors):
        for tensor in tensors:
            self = self.insert_tensor(tensor)
        return self

    def insert_tensor(self, tensor):
        for dim in generic_dims(tensor):
            self = self.insert_dim(dim)
        return self

    def __getitem__(self, dim):
        return self.dim2name[dim]

    def __contains__(self, item):
        return item in self.dim2name

    def keys(self):
        return self.dim2name.keys()

class TorchDimDist():
    def __init__(self, dist_name, *args, **kwargs):
        self.dist_name = dist_name
        param_ndim, self.result_ndim = param_event_ndim[dist_name]
        self.dist = getattr(td, dist_name)
        #convert all args to kwargs, assuming that the arguments in param_event_ndim have the right order
        arg_dict = {argname: args[i] for (i, argname) in enumerate(list(param_ndim.keys())[:len(args)])}
        #Merge args and kwargs into a unified kwarg dict
        dim_args = {**arg_dict, **kwargs}
        #Check for any positional arguments that are also given as a named argument.
        assert len(dim_args) == len(kwargs) + len(arg_dict)

        #There may be unnamed dimensions over which we're batching.
        unnamed_batch_dim_list = tuple(generic_ndim(arg) - param_ndim[argname] for (argname, arg) in dim_args.items())
        assert all(0<=x for x in unnamed_batch_dim_list)
        self.unnamed_batch_dims = max(unnamed_batch_dim_list)

        padded_dim_args = {}
        for (argname, arg) in dim_args.items():
            #Pad all args up to the right lengths.
            padded_dim_args[argname] = pad_nones(arg, self.unnamed_batch_dims+param_ndim[argname])

        #Mapping from dims to names, dealing with any duplicate names in dims
        self.dim2name = Dim2Name().insert_tensors(dim_args.values())
        #A unified list of dims and names
        self.aligned_dims  = list(self.dim2name.keys())
        self.aligned_names = [self.dim2name[dim] for dim in self.aligned_dims]

        #Convert:
        # unaligned named
        # aligned named
        # aligned unnamed
        self.args = {} 
        for (argname, arg) in padded_dim_args.items():
            arg_dims = generic_dims(arg)
            arg_names = [self.dim2name[dim] for dim in arg_dims]
            unaligned_named_arg = generic_order(arg, arg_dims).rename(*arg_names, ...)
            aligned_named_arg = unaligned_named_arg.align_to(*self.aligned_names, ...)
            self.args[argname] = aligned_named_arg.rename(None)

    def sample(self, reparam, sample_dims):
        torch_dist = self.dist(**self.args)
        sample_method = getattr(torch_dist, "rsample" if reparam else "sample")
        sample_dims = set(sample_dims).difference(self.aligned_dims)
        sample_shape = [dim.size for dim in sample_dims]
        sample = sample_method(sample_shape=sample_shape)
        dims = [*sample_dims, *self.aligned_dims, Ellipsis]
        return sample[dims]

    def log_prob(self, x):
        dim2name = self.dim2name.insert_tensor(x)
        x_dims = generic_dims(x)
        unaligned_names = [dim2name[dim] for dim in x_dims]
        unaligned_named = generic_order(x, x_dims).rename(*unaligned_names, ...)

        aligned_dims  = [*set(x_dims).difference(self.aligned_dims), *self.aligned_dims]
        aligned_names = [dim2name[dim] for dim in aligned_dims]
        aligned_x = unaligned_named.align_to(*aligned_names, ...)

        return self.dist(**self.args).log_prob(aligned_x.rename(None))[aligned_dims]

def set_dist(dist_name):
    def inner(*args, **kwargs):
        return TorchDimDist(dist_name, *args, **kwargs)
    globals()[dist_name] = inner

for dist_name in param_event_ndim:
    set_dist(dist_name)


if __name__ == "__main__":
    i = dims(1)
    j = dims(1, [10])
    mean = t.ones(3,3)[i]
    std = t.ones(())
    dist = Normal(mean, std)
    result = dist.sample(False, sample_dims=(j,))
    print(result)

    print(dist.log_prob(result))

    i = dims(1)
    j = dims(1, [10])
    mean = t.ones(3,3)[i]
    cov = t.eye((3))
    dist = MultivariateNormal(mean, precision_matrix=cov)
    result = dist.sample(False, sample_dims=(j,))
    print(result)

    print(dist.log_prob(result))
