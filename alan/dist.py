import torch as t
import torch.distributions as td
from functorch.dim import dims, Tensor
from .TruncatedNormal import TruncatedNormal
from alan.utils import *

import numbers

def univariate(*names):
    return ({name: 0 for name in names}, 0)
univariate_loc_scale = univariate("loc", "scale")

param_event_ndim = {
    "Bernoulli":                 univariate("probs", "logits"),#
    "Beta":                      univariate("concentration1", "concentration0"),#
    "Binomial":                  univariate("total_count", "probs", "logits"),#
    "Categorical":               ({"probs": 1, "logits": 1}, 0),#
    "Cauchy":                    univariate_loc_scale,#
    "Chi2":                      univariate("df"),#
    "ContinuousBernoulli":       univariate("probs", "logits"),#
    "Dirichlet":                 ({"concentration": 1}, 1),#
    "Exponential":               univariate("rate"),#
    "FisherSnedecor":            univariate("df1", "df2"),#
    "Gamma":                     univariate("concentration", "rate"),#
    "Geometric":                 univariate("probs", "logits"),#
    "Gumbel":                    univariate_loc_scale,#
    "HalfCauchy":                univariate("scale"),#
    "HalfNormal":                univariate("scale"),#
    "Kumaraswamy":               univariate("concentration1", "concentration0"),#
    "LKJCholesky":               ({"dim":0, "concentration":0}, 2),#
    "Laplace":                   univariate_loc_scale,#
    "LogNormal":                 univariate_loc_scale,#
    "LowRankMultivariateNormal": ({"loc":1, "cov_factor":2, "cov_diag": 1}, 1),#
    "Multinomial":               ({"total_count": 0, "probs": 1, "logits": 1}, 1),#
    "MultivariateNormal":        ({"loc": 1, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 1}, 1),
    "NegativeBinomial":          univariate("total_count", "probs", "logits"),#
    "Normal":                    univariate_loc_scale,#
    "Pareto":                    univariate("scale", "alpha"),
    "Poisson":                   univariate("rate"),#
    "RelaxedBernoulli":          univariate("temperature", "probs", "logits"),
    #"LogitRelaxedBernoulli":     univariate("temperature", "probs", "logits"),
    "RelaxedOneHotCategorical":  ({"temperature": 0, "probs": 1, "logits": 1}, 0),
    "StudentT":                  univariate("df", "loc", "scale"),
    "TruncatedNormal":           univariate("loc", "scale", "a", "b"),
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

def convert_scalar_args(args):
    for k,v in args.items():
        if isinstance(v, float):
            args[k] = t.tensor(v,dtype=t.float32).reshape((1,))
    return args


class TorchDimDist():
    r"""
    Wrapper for PyTorch dists to make them accept TorchDim arguments.

    :class:`TorchDimDist` allows for sampling (or evaluating the log probability of) TorchDim-ed tensors
    from distributions with non-dimmed arguments as well as sampling from distributions with dimmed arguments


    .. warning::
    self.dist and self.dims are exposed!
    """
    def __init__(self, *args, extra_log_factor=lambda x: 0, **kwargs):
        r"""
        Creates a TorchDimDist.

        Args:
            args (List): *List* of arguments for the underlying PyTorch dist, should correspond to the order of arguments in param_event_ndim
            extra_log_factor (function): (*Optional*) Should be a function mapping from sample to *scalar*
                                         corresponding to an extra term added to the evaluated log probability
            kwargs (Dict): *Dict* of keyword arguments for the underlying PyTorch Dist
        """
        self.extra_log_factor = extra_log_factor
        param_ndim, self.result_ndim = param_event_ndim[self.dist_name]
        for kwarg in kwargs:
            if kwarg not in param_ndim:
                raise Exception(f'Unrecognised argument "{kwarg}" given to "{self.dist_name}" distribution.  "{self.dist_name}" only accepts {tuple(param_ndim.keys())}.')
        #self.dist = getattr(td, dist_name)
        #convert all args to kwargs, assuming that the arguments in param_event_ndim have the right order
        arg_dict = {argname: args[i] for (i, argname) in enumerate(list(param_ndim.keys())[:len(args)])}
        #Merge args and kwargs into a unified kwarg dict
        #self.dim_args = convert_scalar_args({**arg_dict, **kwargs})
        self.dim_args = {**arg_dict, **kwargs}
        #Check for any positional arguments that are also given as a named argument.
        assert len(self.dim_args) == len(kwargs) + len(arg_dict)

        self.dims  = unify_dims(self.dim_args.values())

        #Find out the number of unnamed dims over which we batch.
        unnamed_batch_dims = []
        for (argname, arg) in self.dim_args.items():
            unnamed_batch_dims.append(generic_ndim(arg) - param_ndim[argname])

        assert all(0<=x for x in unnamed_batch_dims)
        self.unnamed_batch_dims = max(unnamed_batch_dims)

        self.all_args = {}
        for (argname, arg) in self.dim_args.items():
            #Pad all args up to the right lengths, so that unnamed batching works.
            arg = pad_nones(arg, self.unnamed_batch_dims+param_ndim[argname])
            #Convert torchdim arguments into aligned tensor arguments.
            arg = singleton_order(arg, self.dims)
            assert not is_dimtensor(arg)
            self.all_args[argname] = arg

    def sample(self, reparam, sample_dims, Kdim=None):
        r"""
        Generates a sample with sample_dims + self.dims dimensions

        Args:
            reparam (bool): *True* for reparameterised sampling (Not supported by all dists)
            sample_dims (List): *List* of dimensions to sample (TorchDim dimensions have corresponding sizes)

        Returns:
            sample (TorchDim.Tensor): sample with correct dimensions
        """
        torch_dist = self.dist(**self.all_args)
        if reparam and not torch_dist.has_rsample:
            raise Exception(f'Trying to do reparameterised sampling of {self.dist_name}, which is not implemented by PyTorch (likely because {self.dist_name} is a distribution over discrete random variables).')
        sample_method = getattr(torch_dist, "rsample" if reparam else "sample")
        sample_dims = set(sample_dims).difference(self.dims)
        sample_shape = [dim.size for dim in sample_dims]
        sample = sample_method(sample_shape=sample_shape)
        dims = [*sample_dims, *self.dims, Ellipsis]
        return sample[dims]

    def log_prob(self, x):
        r"""
        Evaluates the log probability for a sample *x*
        """
        #Same number of unnamed batch dims
        assert x.ndim == self.result_ndim + self.unnamed_batch_dims  #or x.ndim == self.result_ndim + self.unnamed_batch_dims + 1
        #if not (x.ndim == self.result_ndim + self.unnamed_batch_dims):
        #    breakpoint()
        x_dims = generic_dims(x)
        new_dims = [dim for dim in x_dims if (dim not in set(self.dims))]
        all_dims = [*new_dims, *self.dims, Ellipsis]
        log_prob = self.dist(**self.all_args).log_prob(singleton_order(x, all_dims))[all_dims]

        if self.unnamed_batch_dims > 0:
            log_prob = log_prob.sum()

        return log_prob + self.extra_log_factor(x)

    def log_prob_P(self, x, Kdim):
        return self.log_prob(x)

for dn in param_event_ndim:
    if dn == 'TruncatedNormal':
        globals()[dn] = type(dn, (TorchDimDist,), {'dist_name': dn, 'dist': TruncatedNormal})
    else:
        globals()[dn] = type(dn, (TorchDimDist,), {'dist_name': dn, 'dist': getattr(td, dn)})


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
