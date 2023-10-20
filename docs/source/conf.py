# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Alan'
copyright = '2023, Laurence Aitchison, Thomas Heap, Sam Bowyer'
author = 'Laurence Aitchison, Thomas Heap, Sam Bowyer'
release = '0.1'
import sphinx_rtd_theme

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import torch
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    if name == "__call__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
    
extensions = ["myst_parser",
              'sphinx.ext.autodoc', 
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon', 
              'sphinx_math_dollar',
              'sphinx.ext.mathjax',
              "sphinx.ext.intersphinx",  
              "sphinx.ext.todo",
              "sphinx.ext.viewcode",  #
              "sphinx.ext.githubpages",
              "sphinx.ext.ifconfig",
              "nbsphinx"]

templates_path = ['_templates']
exclude_patterns = [".ipynb_checkpoints"]


add_module_names = False

pygments_style = "sphinx"

master_doc = "index"


# extend timeout
nbsphinx_timeout = 120

autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "navigation_depth": 3,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "opt_einsum": ("https://optimized-einsum.readthedocs.io/en/stable/", None),
}