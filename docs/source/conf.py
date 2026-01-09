# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
import shutil
from pathlib import Path

from sphinx_markdown_builder import MarkdownBuilder

# import wavesim.engine.functions

# path setup (relevant for both local and read-the-docs builds)
docs_source_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(docs_source_dir))
sys.path.append(docs_source_dir)
sys.path.append(root_dir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "wavesim"
copyright = "2024, Ivo Vellekoop, and Swapnil Mache, University of Twente"
# author = "Swapnil Mache, Ivo M. Vellekoop"
release = "0.2.0a1"
html_title = "WaveSim - A Python package for wave propagation simulation"

# -- latex configuration -----------------------------------------------------
latex_elements = {
    "preamble": r"""
        \usepackage{authblk}
     """,
    "maketitle": r"""
        \author[1,2]{Swapnil~Mache}
        \author[1*]{Ivo~M.~Vellekoop} 
        \affil[1]{University of Twente, Biomedical Photonic Imaging, TechMed Institute, P. O. Box 217, 7500 AE Enschede, The Netherlands}
        \affil[2]{Previously at: Rayfos Ltd., Winton House, Winton Square, Basingstoke, United Kingdom}
        \affil[*]{Corresponding author: i.m.vellekoop@utwente.nl}
        \publishers{%
            \normalfont\normalsize%
            \parbox{0.8\linewidth}{%
                \vspace{0.5cm}
                The modified Born series (MBS) method is a fast and accurate method for simulating wave 
                propagation in complex structures. The major limitation of MBS is that the size of the structure 
                is limited by the working memory of a single computer or graphics processing unit (GPU). 
                Through this package, we present a domain decomposition method that removes this limitation. We 
                decompose large problems over subdomains while maintaining the accuracy, memory efficiency, and guaranteed monotonic convergence of the method. With this work, we have been able to obtain a 
                factor of $1.95$ increase in size over the single-domain MBS simulations without domain decomposition through a 3D simulation using 2 GPUs. For the Helmholtz problem, we solved a complex structure of size $320 \times 320 \times 320$ wavelengths in just 45 minutes on a dual-GPU system.
            }
        }
        \maketitle
    """,
    "tableofcontents": "",
    "makeindex": "",
    "printindex": "",
    "figure_align": "",
    "extraclassoptions": "notitlepage",
}
latex_docclass = {
    "manual": "scrartcl",
    "howto": "scrartcl",
}
latex_documents = [("index_latex",
                    "wavesim.tex",
                    "WaveSim - A Python package for wave propagation simulation",
                    "",
                    "howto",)]
latex_toplevel_sectioning = "section"
bibtex_default_style = "unsrt"
bibtex_bibfiles = ["references.bib"]
numfig = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    "sphinx_markdown_builder",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "acknowledgements.rst", "sg_execution_times.rst"]
master_doc = ""
include_patterns = ["**"]
napoleon_use_rtype = False
napoleon_use_param = True
typehints_document_rtype = False
latex_engine = "xelatex"
add_module_names = False
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"

# -- Options for sphinx-gallery ----------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "ignore_pattern": "__init__.py|boundary_approximation.py",  # ignore files that match this pattern
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}


# -- Monkey-patch the MarkdownTranslator class to support citations ------------
def visit_citation(self, node):
    """Patch-in function for markdown builder to support citations."""
    id = node["ids"][0]
    # wavesim.engine.functions.add(f'<a name="{id}"></a>')
    self.add(f'<a name="{id}"></a>')


def visit_label(self, node):
    """Patch-in function for markdown builder to support citations."""
    pass


def setup(app):
    """Setup function for the Sphinx extension."""
    # register event handlers
    # app.connect("autodoc-skip-member", skip)
    app.connect("build-finished", build_finished)
    app.connect("builder-inited", builder_inited)
    app.connect("source-read", source_read)
    # # app.connect("autodoc-skip-member", skip)
    # app.connect("build-finished", copy_readme)
    # app.connect("builder-inited", builder_inited)
    # app.connect("source-read", source_read)

    # monkey-patch the MarkdownTranslator class to support citations
    # TODO: this should be done in the markdown builder itself
    cls = MarkdownBuilder.default_translator_class
    # cls.visit_citation = visit_citation
    # cls.visit_label = visit_label


def source_read(app, docname, source):
    """Modify the source of the readme and conclusion files based on the builder."""
    if docname == "readme" or docname == "conclusion":
        if (app.builder.name == "latex") == (docname == "conclusion"):
            source[0] = source[0].replace("%endmatter%", ".. include:: acknowledgements.rst")
        else:
            source[0] = source[0].replace("%endmatter%", "")


def builder_inited(app):
    """Set the master document and exclude patterns based on the builder."""
    if app.builder.name == "html":
        exclude_patterns.extend(["conclusion.rst", "index_latex.rst", "index_markdown.rst"])
        app.config.master_doc = "index"
    elif app.builder.name == "latex":
        exclude_patterns.extend(["auto_examples/*", "index_markdown.rst", "index.rst", "api*"])
        app.config.master_doc = "index_latex"
    elif app.builder.name == "markdown":
        include_patterns.clear()
        include_patterns.extend(["readme.rst", "index_markdown.rst"])
        app.config.master_doc = "index_markdown"


def build_finished(app, exception):
    if exception:
        return

    if app.builder.name == "markdown":
        # Copy the readme file to the root of the documentation directory.
        source_file = Path(app.outdir) / "readme.md"
        destination_dir = Path(app.confdir).parents[1] / "README.md"
        shutil.copy(source_file, destination_dir)

    elif app.builder.name == "latex":
        # The latex builder adds an empty author field to the title page.
        # This code removes it.
        # Define the path to the .tex file
        tex_file = Path(app.outdir) / "wavesim.tex"

        # Read the file
        with open(tex_file, "r") as file:
            content = file.read()

        # Remove \author{} from the file
        content = content.replace(r"\author{}", "")

        # Write the modified content back to the file
        with open(tex_file, "w") as file:
            file.write(content)
