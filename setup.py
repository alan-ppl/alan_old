from setuptools import setup, find_packages

setup(
    name = "tpp",
    version = "0.0.1",
    keywords = ("test", "xxx"),
    license = "MIT Licence",

    packages = find_packages(),
    install_requires=[
    'opt-einsum',
    'tqdm'
    ],
    include_package_data = True,
    platforms = "any",
)