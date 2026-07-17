"""LIBERO benchmark + observation plumbing (controlled slice).

Ported from smolvla_eval_core.py: _quat2axisangle, obs_to_policy, init_libero_benchmark,
build_final_episodes, and the OffScreenRenderEnv factory. LIBERO-PRO suite/episode building
lives in libero_pro.py.
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch

from .config import CAMERAS, IMG_SIZE, MAX_STEPS_MAP

# Set by init_libero_benchmark(); also accepted explicitly by builders for testability.
BENCHMARK_DICT = None

# Controlled 80-episode slice: 8 stock tasks x 10 episodes.
FALLBACK_SLICE_TASKS = [
    ("libero_spatial", 5), ("libero_spatial", 8),
    ("libero_goal", 0), ("libero_goal", 1), ("libero_goal", 2),
    ("libero_goal", 3), ("libero_goal", 5), ("libero_goal", 6),
]


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_policy(obs_dict, task_desc):
    agentview = np.ascontiguousarray(obs_dict["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs_dict["robot0_eye_in_hand_image"][::-1, ::-1])
    img_agent = torch.from_numpy(agentview / 255.0).permute(2, 0, 1).float()
    img_wrist = torch.from_numpy(wrist / 255.0).permute(2, 0, 1).float()
    state = np.concatenate([
        obs_dict["robot0_eef_pos"],
        _quat2axisangle(obs_dict["robot0_eef_quat"]),
        obs_dict["robot0_gripper_qpos"],
    ])
    return {
        "observation.images.image": img_agent,
        "observation.images.image2": img_wrist,
        "observation.state": torch.from_numpy(state).float(),
        "task": task_desc,
    }


def init_state_hash(init_state) -> str:
    return hashlib.md5(np.asarray(init_state).tobytes()).hexdigest()[:12]


def init_libero_benchmark(hf_login: bool = True):
    """Load the LIBERO benchmark dict (+ optional HF login). Returns benchmark_dict."""
    global BENCHMARK_DICT
    from libero.libero import benchmark
    if hf_login:
        from huggingface_hub import login
        tok = os.getenv("HF_TOKEN")
        login(token=tok) if tok else None
    BENCHMARK_DICT = benchmark.get_benchmark_dict()
    print(f"LIBERO ready ({len(BENCHMARK_DICT)} suites)")
    return BENCHMARK_DICT


def make_env(bddl_path: str):
    """OffScreenRenderEnv factory with the standard camera/render config."""
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_names=CAMERAS,
        camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
        has_offscreen_renderer=True, use_camera_obs=True,
        has_renderer=False, reward_shaping=False,
    )


def build_final_episodes(benchmark_dict=None, episode_idxs=None, tasks=None):
    """Build the controlled slice episode list (8 stock tasks x 10 episodes by default)."""
    from libero.libero import get_libero_path
    bd = benchmark_dict or BENCHMARK_DICT
    if bd is None:
        raise RuntimeError("init_libero_benchmark() first (or pass benchmark_dict).")
    episode_idxs = episode_idxs or list(range(10))
    tasks = tasks or FALLBACK_SLICE_TASKS

    episodes = []
    for suite, task_idx in tasks:
        task_suite = bd[suite]()
        task = task_suite.get_task(task_idx)
        init_states = task_suite.get_task_init_states(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        max_steps = MAX_STEPS_MAP.get(suite, 300)
        for ep_idx in episode_idxs:
            if ep_idx >= len(init_states):
                continue
            init_state = init_states[ep_idx]
            episodes.append(dict(
                benchmark="libero", suite=suite, task_idx=task_idx, task_desc=task.language,
                ep_idx=ep_idx, init_state=init_state, bddl_path=bddl_path,
                max_steps=max_steps, init_state_hash=init_state_hash(init_state),
                suite_family="base", perturb_axis=None, perturb_strength=None,
                distractor_object=None,
            ))
    print(f"controlled slice episodes: {len(episodes)}")
    return episodes
