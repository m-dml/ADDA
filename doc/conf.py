# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../adda"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'adda'
copyright = "2026, m-dml team"
author = "m-dml team"
release = "0.1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    #"seed_intersphinx_mapping",
    "sphinx_typo3_theme",
]

add_module_names = True
toc_object_entries_show_parents = "hide"
templates_path = ["_templates"]
exclude_patterns = ["../adda/_version.py"]
autosummary_generate = True

# --- Autodoc ---------------------------------------------------------------


autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "undoc-members": True,
    "exclude-members": "__weakref__, __init__",
    "show-inheritance": True,
}

autoclass_content = "class"
ignore_module_all = False
autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
