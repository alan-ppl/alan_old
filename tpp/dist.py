import torch as t
import torch.distributions as td
from functorch.dim import dims, Tensor
from tpp.utils import *

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



class TorchDimDist():
    def __init__(self, dist_name, *args, **kwargs):
        self.dist_name = dist_name
        param_ndim, self.result_ndim = param_event_ndim[dist_name]
        self.dist = getattr(td, dist_name)
        #convert all args to kwargs, assuming that the arguments in param_event_ndim have the right order
        arg_dict = {argname: args[i] for (i, argname) in enumerate(list(param_ndim.keys())[:len(args)])}
        #Merge args and kwargs into a unified kwarg dict
        self.all_args = {**arg_dict, **kwargs}
        #Check for any positional arguments that are also given as a named argument.
        assert len(self.all_args) == len(kwargs) + len(arg_dict)

        self.dims  = unify_dims(self.all_args.values())

        #Find out the number of unnamed dims over which we batch.
        unnamed_batch_dims = []
        for (argname, arg) in self.all_args.items():
            unnamed_batch_dims.append(generic_ndim(arg) - param_ndim[argname])
        assert all(0<=x for x in unnamed_batch_dims)
        self.unnamed_batch_dims = max(unnamed_batch_dims)


        for (argname, arg) in self.all_args.items():
            #Pad all args up to the right lengths, so that unnamed batching works.
            arg = pad_nones(arg, self.unnamed_batch_dims+param_ndim[argname])
            #Convert torchdim arguments into aligned tensor arguments.
            arg = singleton_order(arg, self.dims)
            assert not is_dimtensor(arg)
            self.all_args[argname] = arg

    def sample(self, reparam, sample_dims):
        torch_dist = self.dist(**self.all_args)
        sample_method = getattr(torch_dist, "rsample" if reparam else "sample")
        sample_dims = set(sample_dims).difference(self.dims)
        sample_shape = [dim.size for dim in sample_dims]
        sample = sample_method(sample_shape=sample_shape)
        dims = [*sample_dims, *self.dims, Ellipsis]
        return sample[dims]

    def log_prob(self, x):
        #Same number of unnamed batch dims.
        assert x.ndim == self.result_ndim + self.unnamed_batch_dims
        x_dims = generic_dims(x)
        new_dims = [dim for dim in x_dims if (dim not in set(self.dims))]
        all_dims = [*new_dims, *self.dims, Ellipsis]
        log_prob = self.dist(**self.all_args).log_prob(singleton_order(x, all_dims))[all_dims]

        if self.unnamed_batch_dims == 0:
            return log_prob
        return log_prob.sum()
        #if len(all_dims) > 1:
        #    return self.dist(**self.all_args).log_prob(singleton_order(x, all_dims))[all_dims].sum().order(*all_dims[:-1])[all_dims[:-1]]
        #else:

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
