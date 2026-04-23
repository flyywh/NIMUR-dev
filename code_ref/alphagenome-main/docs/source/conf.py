# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder."""

#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib.metadata
import inspect
import os
import sys

# The package is installed by Readthedocs before sphinx building.
import alphagenome  # pylint: disable=unused-import, g-import-not-at-top
import alphagenome.models.dna_client  # pylint: disable=unused-import, g-import-not-at-top

# -- Project information -----------------------------------------------------

project = 'alphagenome'
project_info = importlib.metadata.metadata(project)
author = project_info['Author']
copyright = f'2024, {author}'  # pylint: disable=redefined-builtin
version = project_info['Version']
repository_url = f'https://github.com/google-deepmind/{project}'


# The full version, including alpha/beta/rc tags
release = version
# Warn if links are broken
nitpicky = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.coverage',
    'sphinx_copybutton',
    'sphinx_remove_toctrees',
    'sphinx.ext.linkcode',
]

autosummary_generate = True
autodoc_member_order = 'groupwise'
default_role = 'literal'
bibtex_reference_style = 'author_year'
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6  # Create heading anchors for h1-h6
autodoc_mock_imports = [
    'google.protobuf.runtime_version',
    'google.protobuf.internal.builder',
    'absl',
    'alphagenome.protos',
]
remove_from_toctrees = ['api/generated/*']
bibtex_bibfiles = ['refs.bib']

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
    'html_admonition',
    'attrs_inline',
    'attrs_block',
]

# TODO(b/372225132): Resolve showing notebook output without executing.
# TODO(b/372226231): Resolve not modifying notebook when building docs.
myst_url_schemes = ['http', 'https', 'mailto']
nb_output_stderr = 'remove'
nb_execution_mode = 'off'
nb_merge_streams = True
typehints_defaults = 'braces'

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'protos']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}


# -- Source code links -----------------------------------------------------


def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  path = os.path.relpath(filename, start=os.path.dirname(alphagenome.__file__))
  return (
      f'{repository_url}/tree/main/src/{project}/'
      f'{path}#L{lineno}#L{lineno + len(source) - 1}'
  )


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_book_theme'
html_title = 'AlphaGenome'
pygments_style = 'default'
html_theme_options = {
    'repository_url': repository_url,
    'repository_branch': 'main',
    'use_repository_button': True,
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
    },
    'article_header_start': ['toggle-primary-sidebar.html', 'breadcrumbs'],
    'show_prev_next': False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# TODO: b/377291190 - Look at adding notebook support (see haiku example)
