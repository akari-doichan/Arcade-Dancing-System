# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import os
import sys
import sphinx_theme
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'Arcade Dance System'
copyright = '2021, akari-doichan'
author = 'akari-doichan'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxext.opengraph',         # pip install sphinxext-opengraph
    'sphinx.ext.autodoc',          # Include documentation from docstrings.
    'sphinx.ext.autosummary',      # Generate autodoc summaries.
    'sphinx.ext.viewcode',         # Add links to highlighted source code.
    'sphinx.ext.todo',             # Support for todo items.
    'sphinx.ext.napoleon',         # Support for NumPy and Google style docstrings
    'sphinx.ext.githubpages',      # Publish HTML docs in GitHub Pages
    'sphinx.ext.autosectionlabel', # Allow reference sections using its title
    'matplotlib.sphinxext.plot_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'stanford_theme'
html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
html_show_sphinx = False
html_favicon = "./_images/favicon.png"
stickysidebar = True
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# How to sort.
autodoc_member_order = "bysource"

# -- Options for extension ----------------------------------------------

# [todo]
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# [opengraph]
ogp_site_url = "https://akari-doichan.github.io/Arcade-Dancing-System/"
ogp_image = "https://akari-doichan.github.io/Arcade-Dancing-System/_images/header.png"
ogp_description_length = 200
ogp_type = "article"
ogp_custom_meta_tags = [
    '<meta property="og:site_name" content="Arcade-Dancing-System" />',
    '<meta name="twitter:site" content="@akari_doichan" />',
    '<meta name="twitter:card" content="summary">',
    f'<meta name="twitter:image:src" content="{ogp_image}">',
]

# [matplotlib]
plot_include_source         = True   # Default value for the include-source option.
plot_html_show_source_link  = False  # Whether to show a link to the source in HTML.
plot_html_show_formats      = False  # Whether to show links to the files in HTML.