To get startted:

```
pip install -e ./
```


Dependency:
- Torch nightly
- Functorch: https://github.com/facebookresearch/functorch

Notes:
- approximate posterior should be independent of data
- On MacOs, you probably need to `export MACOSX_DEPLOYMENT_TARGET=10.9` before installing functorch

TODOs:
- document that you have to be _really_ careful with dimensions in your programme.
- document how to set the dimensions for data!
- More rigorous testing workflow and cases. (e.g. Using unit test framework like pytest)
- More examples for TPP.