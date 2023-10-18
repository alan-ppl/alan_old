Model
=====

Given a generative model $P$ and an approximate posterior $Q$ we define models as:

.. code-block:: python
    model = alan.Model(P, Q())

This handles things such as:
 - Returning samples from the prior and posterior
 - Predictive samples and predictive log likelihood
 - updating parameters for exponential family approximate posteriors (SVI is handled using PyTorch optimizers)
 
.. automodule:: alan.model
   :members:
   :undoc-members:
   :show-inheritance: