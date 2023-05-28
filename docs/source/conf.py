#!/usr/bin/env python3

import os
import subprocess

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    subprocess.run(["doxygen", "-v"])
    subprocess.call('cd ..; doxygen', shell=True)

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'navigation_depth': 4
}

def setup(app):
    app.add_css_file("main_stylesheet.css")

# extensions = ['breathe','sphinx.ext.mathjax']
extensions = ['breathe','sphinxcontrib.katex','sphinxcontrib.contentui']
breathe_projects = { 'bqreg': '../xml' }
templates_path = ['_templates']
html_static_path = ['_static']
source_suffix = '.rst'
master_doc = 'index'
project = 'Bayesian Quantile Regression'
copyright = '2021-2023 Keith O\'Hara'
author = 'Keith O\'Hara'

exclude_patterns = []
highlight_language = 'c++'
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'bqregdoc'
