Exponential Family
==================

We can perform maximum inference on exponential family models which allows for fast fitting of models with many parameters.  The exponential family is a set of distributions which can be written in the form $q(\theta|\eta) = q_{\eta}(\theta) = h(\theta)\exp [ \langle\eta,\phi(\theta)\rangle - A(\eta) ]$.  The parameters $\eta$ are called the natural parameters and the function $\phi(\theta)$ is called the sufficient statistics.  The function $A(\eta)$ is called the log partition function. We can update these natural parameters efficiently, and these classes provide a way to do this.  

alan.ml module
--------------

.. automodule:: alan.ml
   :members:
   :undoc-members:
   :show-inheritance:

alan.ml2 module
---------------

.. automodule:: alan.ml2
   :members:
   :undoc-members:
   :show-inheritance:


alan.ng module
---------------

.. automodule:: alan.ng
   :members:
   :undoc-members:
   :show-inheritance:

alan.exp\_fam\_mixin module
---------------------------

.. automodule:: alan.exp_fam_mixin
   :members:
   :undoc-members:
   :show-inheritance: