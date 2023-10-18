We provide a set of tools for sampling and inference in Bayesian models. In particular, Alan works in the following way:

1. You define a generative model
2. You define a set of observations, either drawn from the generative model or real data
3. You define a approximate posterior distribution
4. You iteratively draw samples from the approximate posterior distribution and use these to update the parameters of the approximate posterior distribution with an inference algorithm (e.g. variational inference, MCMC, fast fitting for Exponential family posterior etc.)
5. You evaluate inference using predictive log likelihood or posterior predictive checks

The alan.Sample module is concerned with step 4.