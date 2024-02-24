import os
import sys

# sys.path.insert(0, os.path.abspath('../../src')) # only if import does not work.

from upxo.interfaces.os.pck_mgmt import check_and_install_packages

# Add the project root directory to the Python module search path.
# This allows Sphinx to find your project modules for documentation.
sys.path.insert(0, os.path.abspath('..'))

# Sphinx theme configuration
import sphinx_rtd_theme

# General Sphinx project configuration
project = 'UPXO: UKAEA Poly-XTAL Operations'  # The title of the documentation
copyright = '2024, UKAEA, Dr. Sunil Anandatheertha'  # Copyright notice
author = 'Dr. Sunil Anandatheertha'  # Author name

# The version of the project for which the documentation is being generated.
version = '1.26.1'  # Short version
release = '0.1.0'  # Full release version

# Sphinx extensions that enable various features.
extensions = [
    'sphinx_rtd_theme',  # Theme for the documentation
    'sphinx.ext.autodoc',  # Automatically document from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
]

# Configuration for the Napoleon extension, which parses NumPy and Google style docstrings.
napoleon_google_docstring = True  # Enable support for Google style docstrings
napoleon_numpy_docstring = True  # Enable support for NumPy style docstrings

# The theme to use for HTML pages. In this case, using Read The Docs theme.
html_theme = 'sphinx_rtd_theme'

# File suffixes to consider as source files.
source_suffix = '.rst'  # RestructuredText is the default Sphinx markup language

# The master document is the root document where the documentation starts.
master_doc = 'index'  # Typically this is 'index'

# Directory for static files (like style sheets)
html_static_path = ['_static']  # Relative path to the '_static' directory

# Configuration options for the autodoc Sphinx extension.
autodoc_default_flags = ['show-inheritance']  # Show class inheritance
autodoc_member_order = 'bysource'  # Order members by source order

# Additional reStructuredText (reST) declarations for all files.
# Useful for defining global replacements or adding additional links.
rst_epilog = """
.. |Calculator| replace:: :class:`my_module.Calculator`
"""

# If there are modules that Sphinx cannot import (e.g., due to dependencies
# on external systems), you can mock these modules so Sphinx can continue to
# generate the documentation. Uncomment and list the modules to mock.
# from unittest.mock import MagicMock
# sys.modules['external_dependency'] = MagicMock()
