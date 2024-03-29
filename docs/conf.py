# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'intakeio'
copyright = '2018, Burkhard Hoeckendorf'
author = 'Burkhard Hoeckendorf <burkhard.hoeckendorf@pm.me>'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import sphinx_rtd_theme

extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    # 'autoapi.sphinx',
    'autoapi.extension',
    # 'myst_parser',
    # 'myst_nb',
    'sphinx_rtd_theme',
    'recommonmark',
    'sphinx_markdown_tables',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autoapi_type = 'python'
autoapi_dirs = ['../intake_io']
autoapi_options = ['members', 'undoc-members', 'imported-members', 'show-inheritance', 'show-module-summary']
autoapi_python_class_content = 'both'

# nbsphinx_execute_arguments = [
#     "--InlineBackend.figure_formats={'svg', 'pdf'}",
#     "--InlineBackend.rc=figure.dpi=96",
# ]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev/', None),
    'intake': ('https://intake.readthedocs.io/en/stable/', None)
}
