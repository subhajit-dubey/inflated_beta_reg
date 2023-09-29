# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'inflated_beta_reg'
copyright = '2023, Subhajit Dubey'
author = 'Subhajit Dubey'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.spelling'
]

templates_path = ['_templates']
exclude_patterns = []

import pathlib
import sys

func_path = pathlib.Path(__file__).parents[2]
sys.path.insert(0, func_path.resolve().as_posix())

func_path = pathlib.Path(__file__).parents[2] / 'src'
sys.path.insert(0, func_path.resolve().as_posix())


# nbsphinx
nbsphinx_execute = 'always'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
