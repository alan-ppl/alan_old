import torch as t
import torch.distributions as td
import functorch.dim
from alan.utils import *
Tensor = (t.Tensor, functorch.dim.Tensor)


def pad_nones(arg, ndim):
    """
    Pad with as many unnamed dimensions as necessary to reach ndim
    Ignores torchdim dimensions
    """
    if isinstance(arg, Tensor):
        return generic_getitem(arg, (ndim - arg.ndim)*[None]) 
    else: 
        return arg

def parse(spec, args, kwargs):
    """
    Arguments:
        spec: a list of argument names (strings) consituting the "function definition"
        args: a list of values constituting the positional arguments
        kwargs: an argument name -> value dict, constituting the kwargs
    Does error checking, and
    Returns:
        Dict of argument names -> values.
    """
    if len(spec) < len(args):
        raise Exception(f'Too many arguments given to distribution')

    #Check that all argument names in kwargs are in spec.
    for kwarg in kwargs:
        if kwarg not in spec:
            raise Exception(f'Unrecognised argument "{kwarg}" given to distribution.')

    #convert all args to kwargs, assuming that the arguments in param_event_ndim have the right order
    arg_dict = {spec[i]: arg for (i, arg) in enumerate(args)}


    key_overlap = set(arg_dict.keys()).intersection(kwargs.keys())
    if 0 < len(key_overlap):
        raise Exception(f'Multiple values provided for {key_overlap} arguments in distribution.')

    return {**arg_dict, **kwargs}
    

class TorchDimDist():
    r"""
    Wrapper for PyTorch dists to make them accept TorchDim arguments.

    :class:`TorchDimDist` allows for sampling (or evaluating the log probability of) TorchDim-ed tensors
    from distributions with non-dimmed arguments as well as sampling from distributions with dimmed arguments
    
    Note that at present there is no sample_shape dimension, to do IID sampling over
    new non-torchdim dimensions.  To achieve the same effect, do something like
    ```
    alan.Normal(t.randn(3)[:, None].expand(-1, 4), 1)
    ```

    Also note that log-probabilities returned by these classes sum over all non-torchdim
    dimensions (because these are irrelevant for all of Alan's downstream processing)

    .. warning::
    For people editting the class in future: self.dist and self.dims are exposed!
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
        #param_ndim, self.result_ndim = param_event_ndim[self.dist_name]


        #Dict argument name -> value
        self.dim_args = parse(list(self.param_ndim.keys()), args, kwargs)
        #List of torchdims in arguments
        self.dims  = unify_dims(self.dim_args.values())

        #Find the number of unnamed dims over which we batch.
        self.unnamed_batch_dims = 0
        for (argname, arg) in self.dim_args.items():
            ubd = generic_ndim(arg) - self.param_ndim[argname]
            self.unnamed_batch_dims = max(self.unnamed_batch_dims, ubd)

            #Raise an error e.g. if we require a vector but we get a scalar.
            if ubd < 0:
                raise Exception(f'{argname} in {self} should have dimension {self.param_ndim[argname]}, but actually has dimension {generic_ndim(arg)}')

        self.all_args = {}
        for (argname, arg) in self.dim_args.items():
            #Pad all args up to the right lengths, so that unnamed batching works.
            arg = pad_nones(arg, self.unnamed_batch_dims+self.param_ndim[argname])
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
            sample (Tensor): sample with correct dimensions
        """
        assert_unique_dim_iter(sample_dims, 'sample_dims')
       
        torch_dist = self.dist(**self.all_args)
        if reparam and not torch_dist.has_rsample:
            raise Exception(f'Trying to do reparameterised sampling of {type(self)}, which is not implemented by PyTorch (likely because {type(self)} is a distribution over discrete random variables).')
        sample_method = getattr(torch_dist, "rsample" if reparam else "sample")
        sample_dims = set(sample_dims).difference(self.dims)
        sample_shape = [dim.size for dim in sample_dims]
        sample = sample_method(sample_shape=sample_shape)
        dims = [*sample_dims, *self.dims]
        return generic_getitem(sample, dims)

    def log_prob(self, x, Kdim=None):
        assert isinstance(x, Tensor)

        #Same number of unnamed batch dims
        assert x.ndim == self.result_ndim + self.unnamed_batch_dims  #or x.ndim == self.result_ndim + self.unnamed_batch_dims + 1
        #if not (x.ndim == self.result_ndim + self.unnamed_batch_dims):
        #    breakpoint()
        x_dims = generic_dims(x)
        new_dims = [dim for dim in x_dims if (dim not in set(self.dims))]
        all_dims = [*new_dims, *self.dims]
        x = singleton_order(x, all_dims)
        assert not is_dimtensor(x)
        log_prob = self.dist(**self.all_args).log_prob(x)
        log_prob = generic_getitem(log_prob, all_dims)

        if self.unnamed_batch_dims > 0:
            log_prob = log_prob.sum()

        return log_prob + self.extra_log_factor(x)

    def log_prob_Q(self, x, Kdim=None):
        return self.log_prob(x, Kdim=Kdim)

def new_dist(name, dist, result_ndim, param_ndim):
    """
    This is the function called by external code to add a new distribution
    to Alan.
    Arguments:
        name: string, will become the class name for the distribution.
        result_ndim: minimal number of dimensions for a sample (e.g. 1 for vector samples from a Multivariate Gaussian).
        param_ndim: minimal number of dimensions in each parameter, as a dictionary.
    """
    globals()[name] = type(name, (TorchDimDist,), {'dist': dist, 'param_ndim': param_ndim, 'result_ndim': result_ndim})

def new_torch_dist(name, result_ndim, param_ndim):
    """
    Assumes Alan distribution name matches distribution name in torch.distributions
    """
    new_dist(name, getattr(td, name), result_ndim, param_ndim)

def new_univariate_torch_dist(name, params):
    """
    Assumes all parameters and samples are univariate.
    """
    new_torch_dist(name, 0, {param: 0 for param in params})


def new_locscale_torch_dist(name):
    """
    Assumes a univariate distribution with only loc and scale parameters.
    """
    new_torch_dist(name, 0, {'loc': 0, 'scale': 0})

new_univariate_torch_dist("Bernoulli", ("probs", "logits"))
new_univariate_torch_dist("Beta", ("concentration1", "concentration0"))
new_univariate_torch_dist("Binomial", ("total_count", "probs", "logits"))
new_torch_dist("Categorical", 0, {"probs": 1, "logits": 1})
new_locscale_torch_dist("Cauchy")
new_univariate_torch_dist("Chi2", ("df",))
new_univariate_torch_dist("ContinuousBernoulli", ("probs", "logits"))
new_torch_dist("Dirichlet", 1, {"concentration": 1})
new_univariate_torch_dist("Exponential", ("rate",))
new_univariate_torch_dist("FisherSnedecor", ("df1", "df2"))
new_univariate_torch_dist("Gamma", ("concentration", "rate"))
new_univariate_torch_dist("Geometric", ("probs", "logits"))
new_locscale_torch_dist("Gumbel")
new_univariate_torch_dist("HalfCauchy", ("scale",))
new_univariate_torch_dist("HalfNormal", ("scale",))
new_univariate_torch_dist("Kumaraswamy", ("concentration1", "concentration0"))
new_torch_dist("LKJCholesky", 2, {"dim": 0, "concentration": 0})
new_locscale_torch_dist("Laplace")
new_locscale_torch_dist("LogNormal")
new_torch_dist("LowRankMultivariateNormal", 1, {"loc":1, "cov_factor":2, "cov_diag": 1})
new_torch_dist("Multinomial", 1, {"total_count": 0, "probs": 1, "logits": 1})
new_torch_dist("MultivariateNormal", 1, {"loc": 1, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 1})
new_univariate_torch_dist("NegativeBinomial", ("total_count", "probs", "logits"))
new_locscale_torch_dist("Normal")
new_univariate_torch_dist("Pareto", ("scale", "alpha")),
new_univariate_torch_dist("Poisson", ("rate",))
new_univariate_torch_dist("RelaxedBernoulli", ("temperature", "probs", "logits"))
new_torch_dist("RelaxedOneHotCategorical", 0, {"temperature": 0, "probs": 1, "logits": 1})
new_univariate_torch_dist("StudentT", ("df", "loc", "scale"))
new_univariate_torch_dist("Uniform", ("low", "high"))
new_univariate_torch_dist("VonMises", ("loc", "concentration"))
new_univariate_torch_dist("Weibull", ("scale", "concentration"))
new_torch_dist("Wishart", 2, {"df": 0, "covariance_matrix": 2, "precision_matrix": 2, "scale_tril": 2})
