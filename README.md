[![Documentation Status](https://readthedocs.org/projects/alan-ppl/badge/?version=dev)](http://alan-ppl.readthedocs.io/en/latest/?badge=dev)

[Documentation](https://alan-ppl.readthedocs.io/en/latest/)

<!-- To get started:

```
pip install -e ./
```


Dependency:
- Torch 1.13
- Current version of Functorch: https://github.com/facebookresearch/functorch `pip install functorch`

Notes:
- approximate posterior should be independent of data
- On MacOs, you probably need to `export MACOSX_DEPLOYMENT_TARGET=10.9` before installing functorch

TODOs:
- document that you have to be _really_ careful with dimensions in your programme.
- document how to set the dimensions for data!
- More rigorous testing workflow and cases. (e.g. Using unit test framework like pytest)
- More examples for tpp. -->

Alan: Probabilistic Programming with Massively Parallel Importance Weighting
=====================================================

This library showcases Massively Parallel Importance Weighting in the context of Variational Inference based probabilistic programming. Using Importance Weighted Autoencoder (IWAE) and Reweighted Wake Sleep (RWS) as inference methods, for a graphical model with $n$ latent variables we can obtain $K^n$ proposals where $K$ is determined by the user. This improves inference performance and allows for...

For more information see the [Introduction to Alan guide](https://alan-ppl.readthedocs.io/en/latest/introduction_alan.html)
