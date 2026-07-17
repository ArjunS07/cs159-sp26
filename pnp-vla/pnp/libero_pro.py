"""LIBERO-PRO environment surgery + episode building.

Ported from final/pnp_pro_experiment_averages.ipynb (primary) + 'pnp_pro_experiment copy.ipynb'.
This module MUTATES the installed `libero` package on disk (copies patched files from a cloned
LIBERO-PRO repo, rewrites the benchmark task map, registers position-perturb + distractor
suites) and patches torch.load so LIBERO `.pruned_init` files deserialize. It only runs
meaningfully on a Colab GPU runtime with the LIBERO-PRO assets present; there is nothing to
unit-test off-Colab.

Typical setup order (see the LIBERO-PRO setup notebook):
    clone_libero_pro()                       # git clone Zxy-MLlab/LIBERO-PRO
    apply_env_patches(libero_site())         # copy files + rewrite task map (cell 17)
    patch_torch_load()                        # weights_only=False + quant/dynamo (cell 19)
    # (one-time) generate/restore .pruned_init init states into <libero_site>/init_files
    bd = reload_benchmark()                    # importlib.reload benchmark, get_benchmark_dict()
    eps = build_libero_pro_episodes(bd)        # 600-episode list w/ descriptors
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import sys

from .config import MAX_STEPS_MAP, LIBERO_PRO_MAX_STEPS

LIBERO_PRO_REPO = "https://github.com/Zxy-MLlab/LIBERO-PRO.git"
PRO_SUFFIXES = ("_lan", "_swap", "_object", "_task", "_env", "_temp")

TEMP_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
]
TEMP_STRENGTHS = ([f"libero_object_temp_x{v}" for v in ["0.1", "0.2", "0.3", "0.4", "0.5"]] +
                  [f"libero_object_temp_y{v}" for v in ["0.1", "0.2", "0.3", "0.4", "0.5"]])

# The 6-suite headline PRO set (position-perturb + distractor).
DEFAULT_PRO_SUITES = [
    "libero_object_temp_x0.1", "libero_object_temp_x0.2",
    "libero_object_temp_y0.1", "libero_object_temp_y0.2",
    "libero_spatial_with_milk", "libero_goal_with_yellow_book",
]


def libero_site() -> str:
    return f"/usr/local/lib/python3.{sys.version_info.minor}/dist-packages/libero/libero"


# ─────────────────────────────────────────────────────────────────────────────
# Setup (Colab-only; mutates the installed libero package)
# ─────────────────────────────────────────────────────────────────────────────
def clone_libero_pro(dest: str = "/content/LIBERO-PRO") -> str:
    if not os.path.isdir(dest):
        import subprocess
        subprocess.check_call(["git", "clone", "--depth=1", LIBERO_PRO_REPO, dest])
    return dest


def apply_env_patches(site: str | None = None, pro_dir: str = "/content/LIBERO-PRO") -> None:
    """Copy patched libero files + rewrite the task map (ports averages cell 17)."""
    site = site or libero_site()
    bddl_dir = os.path.join(site, "bddl_files")

    # 1) copy patched files from the LIBERO-PRO repo (backing up originals)
    for rel in ("benchmark/__init__.py", "benchmark/libero_suite_task_map.py",
                "envs/objects/__init__.py"):
        src = os.path.join(pro_dir, "libero/libero", rel)
        dst = os.path.join(site, rel)
        if not os.path.exists(src):
            print(f"WARNING: not found in clone: {rel}"); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        bak = dst + ".original_bak"
        if os.path.exists(dst) and not os.path.exists(bak):
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"Patched: {rel}")
    # copy any new object .py files
    src_obj = os.path.join(pro_dir, "libero/libero/envs/objects")
    dst_obj = os.path.join(site, "envs/objects")
    if os.path.isdir(src_obj):
        for f in os.listdir(src_obj):
            if f.endswith(".py") and not os.path.exists(os.path.join(dst_obj, f)):
                shutil.copy2(os.path.join(src_obj, f), os.path.join(dst_obj, f))
                print(f"Added new object file: {f}")

    binit = os.path.join(site, "benchmark/__init__.py")
    content = open(binit).read()
    # 2) libero_mine KeyError -> .get()
    content = content.replace("for task in libero_task_map[libero_suite]:",
                              "for task in libero_task_map.get(libero_suite, []):")
    # 3) make libero_suites dynamic
    if "libero_suites = list(libero_task_map.keys())" not in content:
        m = re.search(r"libero_suites\s*=\s*[\[\(].*?[\]\)]", content, re.DOTALL)
        if m:
            content = content[:m.start()] + "libero_suites = list(libero_task_map.keys())" + content[m.end():]
        else:
            content = content.replace("for libero_suite in libero_suites:",
                                      "libero_suites = list(libero_task_map.keys())\n"
                                      "for libero_suite in libero_suites:")
    open(binit, "w").write(content)

    # 4) + 5) register PRO + position-perturb suites in the task map
    tmap = os.path.join(site, "benchmark/libero_suite_task_map.py")
    existing = open(tmap).read()
    additions = []
    for suite_dir in sorted(os.listdir(bddl_dir)) if os.path.isdir(bddl_dir) else []:
        if not any(suite_dir.endswith(s) for s in PRO_SUFFIXES):
            continue
        tasks = sorted(f[:-5] for f in os.listdir(os.path.join(bddl_dir, suite_dir))
                       if f.endswith(".bddl"))
        if tasks and not re.search(rf'"{re.escape(suite_dir)}"\s*[:\]]\s*\[.*?\S.*?\]',
                                   existing, re.DOTALL):
            additions.append((suite_dir, tasks))
    temp_add = [s for s in TEMP_STRENGTHS if f'"{s}"' not in existing]
    with open(tmap, "a") as f:
        for suite_name, tasks in additions:
            f.write(f'\nlibero_task_map["{suite_name}"] = [\n')
            f.writelines(f'    "{t}",\n' for t in tasks)
            f.write("]\n")
        for suite in temp_add:
            f.write(f'\nlibero_task_map["{suite}"] = [\n')
            f.writelines(f'    "{t}",\n' for t in TEMP_TASKS)
            f.write("]\n")
    print(f"Registered {len(additions)} PRO + {len(temp_add)} position-perturb suites")


def patch_torch_load() -> None:
    """Make torch.load default weights_only=False so LIBERO .pruned_init files load (cell 19)."""
    import numpy as np
    import torch
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        import torch._dynamo
        torch._dynamo.config.disable = True
    except Exception:
        pass
    import torch.ao.quantization as _q
    for name, val in [("CUSTOM_KEY", "custom"), ("NUMERIC_DEBUG_HANDLE_KEY", "numeric_debug_handle")]:
        if not hasattr(_q, name):
            setattr(_q, name, val)
    try:
        torch.serialization.add_safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.core.multiarray.scalar])
    except Exception:
        pass
    if not hasattr(torch, "_libero_true_load"):
        torch._libero_true_load = torch.load
    true = torch._libero_true_load

    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return true(*a, **kw)

    torch.load = _patched
    print("torch.load patched (weights_only=False default).")


def reload_benchmark():
    """importlib.reload the benchmark module after patching, return get_benchmark_dict()."""
    from libero.libero import benchmark as _bm
    importlib.reload(_bm)
    bd = _bm.get_benchmark_dict()
    print(f"benchmark reloaded ({len(bd)} suites)")
    return bd


def restore_libero_pro_inits(init_src: str, site: str | None = None, suites=None) -> None:
    """Copy curated .pruned_init files from init_src/<suite>/ into <site>/init_files/<suite>/."""
    site = site or libero_site()
    dst_root = os.path.join(site, "init_files")
    suites = suites or [d for d in os.listdir(init_src) if os.path.isdir(os.path.join(init_src, d))]
    for suite in suites:
        src = os.path.join(init_src, suite)
        dst = os.path.join(dst_root, suite)
        if not os.path.isdir(src):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith(".pruned_init"):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"restored init files for {len(suites)} suite(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Suite descriptors (explicit columns; no regex-parsing in analysis)
# ─────────────────────────────────────────────────────────────────────────────
def describe_suite(suite: str) -> dict:
    m = re.search(r"_temp_([xy])([\d.]+)$", suite)
    if m:
        return {"suite_family": "position_perturb", "perturb_axis": m.group(1),
                "perturb_strength": float(m.group(2)), "distractor_object": None}
    if suite.endswith("_swap"):
        return {"suite_family": "swap", "perturb_axis": None, "perturb_strength": None,
                "distractor_object": None}
    if suite.endswith("_task"):
        return {"suite_family": "task", "perturb_axis": None, "perturb_strength": None,
                "distractor_object": None}
    m = re.search(r"_with_(.+)$", suite)
    if m:
        return {"suite_family": "distractor", "perturb_axis": None, "perturb_strength": None,
                "distractor_object": m.group(1)}
    return {"suite_family": "base", "perturb_axis": None, "perturb_strength": None,
            "distractor_object": None}


# ─────────────────────────────────────────────────────────────────────────────
# Episode building
# ─────────────────────────────────────────────────────────────────────────────
def build_libero_pro_episodes(benchmark_dict, suites=None, episode_idxs=None):
    """Build the PRO episode list (suites x tasks x episodes) with descriptors."""
    from .libero_env import init_state_hash
    from libero.libero import get_libero_path
    suites = suites or DEFAULT_PRO_SUITES
    episode_idxs = episode_idxs if episode_idxs is not None else list(range(10))
    episodes = []
    for suite in suites:
        desc = describe_suite(suite)
        task_suite = benchmark_dict[suite]()
        max_steps = MAX_STEPS_MAP.get(suite, LIBERO_PRO_MAX_STEPS)
        for task_idx in range(task_suite.n_tasks):
            task = task_suite.get_task(task_idx)
            init_states = task_suite.get_task_init_states(task_idx)
            bddl_path = os.path.join(get_libero_path("bddl_files"),
                                     task.problem_folder, task.bddl_file)
            for ep_idx in episode_idxs:
                if ep_idx >= len(init_states):
                    continue
                init_state = init_states[ep_idx]
                episodes.append(dict(
                    benchmark="libero_pro", suite=suite, task_idx=task_idx,
                    task_desc=task.language, ep_idx=ep_idx, init_state=init_state,
                    bddl_path=bddl_path, max_steps=max_steps,
                    init_state_hash=init_state_hash(init_state), **desc))
    print(f"LIBERO-PRO episodes: {len(episodes)} ({len(suites)} suites)")
    return episodes
