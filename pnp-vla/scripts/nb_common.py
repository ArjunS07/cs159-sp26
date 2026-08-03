"""Shared building blocks for the notebook generators.

One home for the cell constructors and the bootstrap cell, so every generator
emits the same thin setup. The bootstrap cell fetches ``colab_bootstrap.py`` from
raw GitHub (``main``) and ``exec``s it, replacing the ~20-line clone/install block
that used to be copy-pasted into every notebook. Note: because the cell always
pulls from ``main``, ``colab_bootstrap.py`` must be committed and pushed before a
regenerated notebook is run on Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/ArjunS07/cs159-sp26/main/"
    "pnp-vla/scripts/colab_bootstrap.py")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def notebook(cells, name):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": name},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def write_notebook(path, document):
    path.write_text(json.dumps(document, indent=1) + "\n")


def bootstrap(extras: str = "sim,analysis", setup_env: bool = False) -> str:
    """Return the source for a notebook's fetch-and-exec bootstrap cell."""
    return (
        f"EXTRAS = {extras!r}\n"
        f"SETUP_ENV = {setup_env}\n"
        "import urllib.request\n"
        f"exec(urllib.request.urlopen({BOOTSTRAP_URL!r}).read().decode())")


#: Default bootstrap for GPU + analysis notebooks (08-12).
BOOTSTRAP = bootstrap()
