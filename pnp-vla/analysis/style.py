"""Figure styling — one set_style() applied once; plot fns never redefine rcParams.

seaborn + a consistent, colorblind-safe (Okabe-Ito) color-per-method palette, LaTeX serif
font (graceful fallback when no LaTeX toolchain), clean/spacious layout, savefig at dpi=300
(PNG + vector PDF).
"""
from __future__ import annotations

import os

# Canonical method order/labels for consistent colors across every figure.
METHOD_ORDER = [
    "vanilla",
    "extra_steps",
    "pnp_uncertainty_only",
    "pnp_refinement",          # refine_average=False (last)
    "pnp_refinement_avg",      # refine_average=True
    "multi_sample_select",
    "pnp_only",                # PCP lambda=0
    "pcp",                     # PCP lambda>0
]

# Okabe-Ito colorblind-safe palette (8 hues), mapped by method.
_OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PALETTE = {m: _OKABE_ITO[i % len(_OKABE_ITO)] for i, m in enumerate(METHOD_ORDER)}

METHOD_LABELS = {
    "vanilla": "Vanilla", "extra_steps": "Extra steps",
    "pnp_uncertainty_only": "P\\&P (uncertainty)",
    "pnp_refinement": "P\\&P refine (last)",
    "pnp_refinement_avg": "P\\&P refine (avg)",
    "multi_sample_select": "Multi-sample", "pnp_only": "PnP-only", "pcp": "PCP",
}


def set_style(use_latex: bool = True, base_fontsize: int = 11) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid")
    latex_ok = False
    if use_latex:
        from shutil import which
        latex_ok = which("latex") is not None
    mpl.rcParams.update({
        "text.usetex": latex_ok,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (RSS-friendly)
        "ps.fonttype": 42,
    })
    if not latex_ok and use_latex:
        print("[style] LaTeX toolchain not found — using mathtext CM fallback.")


def method_color(method: str) -> str:
    return PALETTE.get(method, "#666666")


def savefig(fig, name: str, out_dir: str = "figures") -> None:
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[style] wrote {out_dir}/{name}.png|pdf")
