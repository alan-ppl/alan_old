To get startted:

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
- More examples for tpp.
