"""Sphinx configuration for quantspt documentation."""

project = "quantspt"
copyright = "2026, Aheli Poddar"
author = "Aheli Poddar"
release = "0.1.0.dev0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"
