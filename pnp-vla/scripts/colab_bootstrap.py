"""Hosted Colab bootstrap: clone the repo, install the package, expose ``pnp``.

Notebooks fetch this from raw GitHub and ``exec()`` it in their global namespace,
so after it runs ``repo_dir``, ``package_dir``, and ``pnp`` are defined in the
notebook. Callers may set two globals before exec:

    EXTRAS    -- pip extras to install, e.g. "sim" or "sim,analysis"  (default "sim,analysis")
    SETUP_ENV -- when True, also run pnp.env_setup.setup_environment() (default False)

This keeps the per-notebook setup cell down to a three-line fetch-and-exec. There
is no import from ``pnp`` here because the package isn't installed until we run.
"""
import os
import subprocess
import sys

EXTRAS = globals().get("EXTRAS", "sim,analysis")
SETUP_ENV = bool(globals().get("SETUP_ENV", False))

try:
    from google.colab import userdata

    for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "HF_TOKEN", "WANDB_API_KEY"):
        try:
            value = userdata.get(key)
        except Exception:  # secret not set in this Colab profile
            value = None
        if value:
            os.environ[key] = value
    repo_dir = "/content/cs159-sp26"
    gh_pat = userdata.get("GH_PAT")
    repo_url = f"https://{gh_pat}@github.com/ArjunS07/cs159-sp26.git"
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "clone", "--branch", "main", repo_url, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only", "origin", "main"], check=True)
except ImportError:  # not on Colab -- run against the local checkout
    repo_dir = os.path.abspath("..") if os.path.basename(os.getcwd()) == "pnp-vla" else os.getcwd()

package_dir = os.path.join(repo_dir, "pnp-vla")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", f"{package_dir}[{EXTRAS}]"],
    check=True)
# The editable install adds a .pth a fresh interpreter would read at startup; this
# interpreter is already running, so expose the source tree immediately.
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

import pnp

if SETUP_ENV:
    from pnp import env_setup

    env_setup.setup_environment()
print("Loaded pnp from:", pnp.__file__)
