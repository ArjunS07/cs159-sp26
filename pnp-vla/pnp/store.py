"""SupabaseStore — provenance-first results store (Postgres + Storage).

RolloutDB-shaped API over supabase-py so drivers barely change. Small uncertainty rows go to
Postgres; bulky arrays (a_hats, PCP obs_enc+z_hat, trajectories, videos, encodings, ckpts) go
to the `artifacts` Storage bucket, linked from the row by *_path columns.

Env: SUPABASE_URL, SUPABASE_SERVICE_KEY. Schema: supabase/schema.sql.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import subprocess
import uuid
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np

from .config import SCHEMA_VERSION, SAMPLER_ALGO_VERSION, PI05_REPO_ID, ADIM

BUCKET = "artifacts"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance gathering
# ─────────────────────────────────────────────────────────────────────────────
def _git(*args, cwd=None) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _pkg_version(mod: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception:
        return ""


def gather_provenance(model_repo_id: str = PI05_REPO_ID, model_revision: str = "",
                      weights_sha256: str = "") -> dict:
    """Best-effort environment/model provenance snapshot (all optional; torch guarded)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prov: dict[str, Any] = {
        "pnp_git_sha": _git("rev-parse", "HEAD", cwd=here),
        "git_dirty": bool(_git("status", "--porcelain", cwd=here)),
        "pnp_version": _pkg_version("pnp"),
        "sampler_algo_version": SAMPLER_ALGO_VERSION,
        "schema_version": SCHEMA_VERSION,
        "policy_model": "pi05",
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "weights_sha256": weights_sha256,
        "lerobot_version": _pkg_version("lerobot"),
        "torch_version": _pkg_version("torch"),
        "libero_version": _pkg_version("libero"),
        "mujoco_gl": os.getenv("MUJOCO_GL", ""),
        "python_version": os.getenv("PYTHON_VERSION", "") or _python_version(),
        "hostname": os.getenv("HOSTNAME", "") or _hostname(),
    }
    try:
        import torch
        prov["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        prov["tf32_enabled"] = bool(torch.backends.cuda.matmul.allow_tf32)
        prov["matmul_precision"] = torch.get_float32_matmul_precision()
        prov["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return prov


def _python_version() -> str:
    import platform
    return platform.python_version()


def _hostname() -> str:
    import socket
    try:
        return socket.gethostname()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────────────────────
def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def _parquet_bytes(df) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────
class SupabaseStore:
    def __init__(self, url: str | None = None, key: str | None = None, bucket: str = BUCKET,
                 encoding_cache_size: int = 256):
        from supabase import create_client
        url = url or os.environ["SUPABASE_URL"]
        key = key or os.environ["SUPABASE_SERVICE_KEY"]
        self.client = create_client(url, key)
        self.bucket = bucket
        self.run_id: str | None = None
        self.experiment: str | None = None
        self._bytes_written = 0
        self._enc_lru: "OrderedDict[str, np.lib.npyio.NpzFile]" = OrderedDict()
        self._enc_lru_size = encoding_cache_size

    # ── hashing / ids ──────────────────────────────────────────────────────
    @staticmethod
    def init_state_hash(init_state) -> str:
        return hashlib.md5(np.asarray(init_state).tobytes()).hexdigest()[:12]

    @staticmethod
    def config_hash(logical_config: dict) -> str:
        return hashlib.sha256(json.dumps(logical_config, sort_keys=True, default=_to_jsonable)
                              .encode()).hexdigest()

    def make_rollout_id(self, experiment: str, identity: dict, logical_config: dict) -> str:
        key = f"{experiment}|{json.dumps(identity, sort_keys=True, default=_to_jsonable)}" \
              f"|{self.config_hash(logical_config)}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    # ── experiment / run lifecycle ─────────────────────────────────────────
    def start_run(self, driver: str, benchmark: str, experiment: str | None = None,
                  notes: str = "", provenance: dict | None = None,
                  config: dict | None = None) -> str:
        experiment = experiment or self._auto_experiment(driver, config)
        self.experiment = experiment
        self.client.table("experiments").upsert(
            {"experiment": experiment}, on_conflict="experiment").execute()
        self.run_id = str(uuid.uuid4())
        row = {
            "run_id": self.run_id, "experiment": experiment, "driver": driver,
            "benchmark": benchmark, "status": "running",
            "label": self._auto_label(driver, config), "notes": notes,
            "config_json": self._json(config or {}),
        }
        row.update(provenance or gather_provenance())
        self.client.table("experiment_runs").insert(row).execute()
        print(f"[store] run {self.run_id[:8]} experiment='{experiment}' driver={driver}")
        return self.run_id

    def finish_run(self, status: str = "completed", n_rollouts: int | None = None) -> None:
        if not self.run_id:
            return
        patch = {"status": status, "finished_at": _now()}
        if n_rollouts is not None:
            patch["n_rollouts"] = n_rollouts
        self.client.table("experiment_runs").update(patch).eq("run_id", self.run_id).execute()

    @staticmethod
    def _auto_experiment(driver: str, config: dict | None) -> str:
        day = _dt.date.today().isoformat()
        summ = ""
        if config:
            summ = "-" + "-".join(str(config[k]) for k in ("suite", "method") if k in config)
        return f"{driver}-{day}{summ}"

    @staticmethod
    def _auto_label(driver: str, config: dict | None) -> str:
        return f"{driver}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ── resume ─────────────────────────────────────────────────────────────
    def existing_keys(self, experiment: str, **eq) -> set[str]:
        q = self.client.table("rollouts").select("rollout_id").eq("experiment", experiment)
        for k, v in eq.items():
            q = q.eq(k, v)
        rows = q.execute().data or []
        return {r["rollout_id"] for r in rows}

    # ── episode logging ────────────────────────────────────────────────────
    def log_episode(self, rollout_row: dict, euler_steps: list[dict] | None = None,
                    action_vectors: list[dict] | None = None,
                    blobs: dict | None = None) -> str:
        """Upsert one rollout (+ its per-step rows + Storage blobs).

        `blobs` may contain: ahats(dict of arrays), trajectory(dict of arrays),
        obs_frames(list of arrays), video(bytes), pcp_chunks(pandas.DataFrame).
        Their Storage keys are written back onto the row.
        """
        rid = rollout_row["rollout_id"]
        rollout_row.setdefault("run_id", self.run_id)
        rollout_row.setdefault("experiment", self.experiment)
        rollout_row.setdefault("sampler_algo_version", SAMPLER_ALGO_VERSION)
        rollout_row.setdefault("schema_version", SCHEMA_VERSION)
        rollout_row["updated_at"] = _now()

        for name, key, payload in self._blob_specs(rid, blobs or {}):
            self._upload(key, payload)
            rollout_row[f"{name}_path"] = key

        self.client.table("rollouts").upsert(self._json(rollout_row),
                                             on_conflict="rollout_id").execute()
        # replace per-step rows for idempotent re-writes
        if euler_steps is not None:
            self.client.table("pnp_euler_steps").delete().eq("rollout_id", rid).execute()
            if euler_steps:
                self.client.table("pnp_euler_steps").insert(
                    [self._json({**s, "rollout_id": rid}) for s in euler_steps]).execute()
        if action_vectors is not None:
            self.client.table("pnp_action_vectors").delete().eq("rollout_id", rid).execute()
            if action_vectors:
                self.client.table("pnp_action_vectors").insert(
                    [self._json({**v, "rollout_id": rid}) for v in action_vectors]).execute()
        return rid

    def _blob_specs(self, rid: str, blobs: dict) -> Iterable[tuple[str, str, bytes]]:
        if blobs.get("ahats"):
            yield "ahats", f"ahats/{rid}.npz", _npz_bytes(blobs["ahats"])
        if blobs.get("trajectory"):
            yield "trajectory", f"trajectories/{rid}.npz", _npz_bytes(blobs["trajectory"])
        if blobs.get("obs_frames"):
            yield "obs_frames", f"obs_frames/{rid}.npz", _npz_bytes(
                {f"f{i}": a for i, a in enumerate(blobs["obs_frames"])})
        if blobs.get("pcp_chunks") is not None:
            yield "pcp_chunks", f"pcp_chunks/{rid}.parquet", _parquet_bytes(blobs["pcp_chunks"])
        if blobs.get("video"):
            yield "video", f"videos/{rid}.mp4", blobs["video"]

    # ── notebook-ergonomic helpers (the driver-loop moved into notebooks) ──
    @staticmethod
    def _denorm(method: str, config) -> dict:
        """Denormalized config columns for filtering + the logical-config hash key."""
        pnp = config.pnp
        on = bool(pnp and pnp.enabled)
        return {
            "method": method,
            "pnp_enabled": on,
            "pnp_step_indices": list(pnp.step_indices) if (on and pnp.step_indices) else None,
            "pnp_k": pnp.num_iterations if on else None,
            "refine_average": pnp.refine_average if on else None,
            "pnp_time_min": pnp.time_min if on else None,
            "num_inference_steps": config.num_inference_steps,
            "num_samples": config.num_samples,
        }

    def rollout_id(self, experiment: str, ep: dict, method: str, config) -> str:
        identity = {k: ep.get(k) for k in
                    ("benchmark", "suite", "task_idx", "episode_idx", "ep_idx", "init_state_hash")}
        return self.make_rollout_id(experiment, identity, self._denorm(method, config))

    @staticmethod
    def _dim_cols(vec, prefix):
        vec = list(vec or [])
        return {f"{prefix}{i}": (float(vec[i]) if i < len(vec) else None) for i in range(ADIM)}

    def _recorder_to_rows(self, rec_ep, chunk_noise_seeds):
        """PnPRecorder episode -> (euler_steps rows, action_vectors rows, summary metrics)."""
        euler, vecs, u_means, u_vecs, mm_bc = [], [], [], [], []
        for c in (rec_ep or {}).get("chunks", []):
            ci = c["chunk_idx"]
            cns = chunk_noise_seeds[ci] if ci < len(chunk_noise_seeds) else None
            for st in c.get("steps", []):
                u_means.append(st["u_mean"]); u_vecs.append(np.asarray(st.get("u_vec", [])))
                euler.append({"chunk_idx": ci, "chunk_noise_seed": cns, "euler_step": st["step"],
                              "s": st["s"], "u_mean": st["u_mean"], "u_max": st["u_max"],
                              "a_std_mean": st.get("a_std_mean"),
                              **self._dim_cols(st.get("u_vec"), "u_d"),
                              **self._dim_cols(st.get("a_std_vec"), "a_std_d")})
                v = {"chunk_idx": ci, "euler_step": st["step"],
                     "a_mean_vec": list(map(float, st.get("a_mean_vec", []))),
                     "a_std_vec": list(map(float, st.get("a_std_vec", [])))}
                if "bc_vec" in st:
                    v.update(bc_vec=list(map(float, st["bc_vec"])),
                             mm_pc1_frac=st.get("mm_pc1_frac"), mm_bc_pc1=st.get("mm_bc_pc1"))
                    mm_bc.append(st.get("mm_bc_pc1"))
                vecs.append(v)
        summary = {"u_mean_episode": float(np.mean(u_means)) if u_means else None,
                   "u_max_episode": float(np.max(u_means)) if u_means else None,
                   "n_pnp_activations": len(euler),
                   "mm_bc_pc1_episode": float(np.nanmean(mm_bc)) if mm_bc else None}
        if u_vecs:
            padded = np.stack([np.pad(u, (0, max(0, ADIM - len(u))))[:ADIM] for u in u_vecs])
            mean_uv = np.nanmean(padded, axis=0)
            summary.update({f"u_mean_d{i}": float(mean_uv[i]) for i in range(ADIM)})
        return euler, vecs, summary

    def log_result(self, rid: str, ep: dict, method: str, config, result: dict) -> str:
        """Map a run_episode result onto the canonical rollouts schema and persist it."""
        euler, vecs, summary = self._recorder_to_rows(result.get("recorder_episode"),
                                                       result.get("chunk_noise_seeds", []))
        denorm = self._denorm(method, config)
        row = {
            "rollout_id": rid,
            "benchmark": ep.get("benchmark"), "suite": ep["suite"], "task_idx": ep["task_idx"],
            "task_desc": ep.get("task_desc"), "episode_idx": ep.get("ep_idx", ep.get("episode_idx")),
            "init_state_hash": ep.get("init_state_hash"),
            "suite_family": ep.get("suite_family"), "perturb_axis": ep.get("perturb_axis"),
            "perturb_strength": ep.get("perturb_strength"),
            "distractor_object": ep.get("distractor_object"),
            "max_steps": ep.get("max_steps"), "chunk_size": result.get("chunk_size"),
            "n_chunks": result["n_chunks"], "action_dim": ADIM,
            "episode_seed": result["episode_seed"], "config_hash": self.config_hash(denorm),
            "config_json": denorm,
            "success": result["success"], "n_steps": result["n_steps"],
            "elapsed_s": result["elapsed_s"],
            "terminated_reason": "success" if result["success"] else result["status"],
            "status": result["status"], "error_msg": result["error_msg"],
            "nan_action_count": result["nan_action_count"], "n_vf_evals": result["n_vf_evals"],
            **denorm, **summary, **result["instability"],
        }
        if config.pcp is not None and config.pcp.mode == "correct":
            row.update(correction_lambda=config.pcp.lambda_pcp, q_gate=config.pcp.q_gate,
                       correction_steps=list(config.pcp.correction_steps),
                       q_ckpt_id=config.pcp.q_ckpt_id, **(result.get("pcp_telemetry") or {}))
        blobs = {}
        if result.get("trajectory"):
            blobs["trajectory"] = result["trajectory"]
        if result.get("obs_frames"):
            blobs["obs_frames"] = result["obs_frames"]
        if config.record_per_iteration and result.get("recorder_episode"):
            ah = {f"c{c['chunk_idx']}_s{st['step']}": st["a_hats"]
                  for c in result["recorder_episode"].get("chunks", [])
                  for st in c.get("steps", []) if "a_hats" in st}
            if ah:
                blobs["ahats"] = ah
        return self.log_episode(row, euler_steps=euler, action_vectors=vecs, blobs=blobs or None)

    def log_collect(self, rid: str, ep: dict, result: dict) -> str:
        """Persist a PCP-collection episode: qc_rollouts row + per-chunk parquet blob."""
        import pandas as pd
        rows = []
        for c in result.get("collected_chunks", []):
            for st in c["steps"]:
                rows.append({"chunk_idx": c["chunk_idx"], "chunk_pos": c["chunk_pos"],
                             "obs_enc": np.asarray(c["obs_enc"]).tolist(),
                             "step_idx": st["step_idx"], "s": st["s"],
                             "z_hat": np.asarray(st["z_hat"]).reshape(-1).tolist()})
        return self.log_qc_rollout(
            {"rollout_id": rid, "suite": ep["suite"], "task_idx": ep["task_idx"],
             "episode_idx": ep.get("ep_idx"), "init_state_hash": ep["init_state_hash"],
             "success": result["success"], "n_chunks": len(result.get("collected_chunks", []))},
            chunks_df=pd.DataFrame(rows) if rows else None)

    def log_eval(self, rid: str, ep: dict, lam: float, result: dict) -> None:
        """Persist a PCP 3-way eval outcome to qc_eval (lam: -1 vanilla / 0 pnp / 3 pcp)."""
        self.log_qc_eval({"rollout_id": rid, "lambda": lam, "suite": ep["suite"],
                          "task_idx": ep["task_idx"], "episode_idx": ep.get("ep_idx"),
                          "success": result["success"]})

    # ── PCP ────────────────────────────────────────────────────────────────
    def log_qc_rollout(self, row: dict, chunks_df=None) -> str:
        rid = row["rollout_id"]
        row.setdefault("run_id", self.run_id)
        row.setdefault("experiment", self.experiment)
        if chunks_df is not None:
            key = f"pcp_chunks/{rid}.parquet"
            self._upload(key, _parquet_bytes(chunks_df))
            row["chunks_path"] = key
        self.client.table("qc_rollouts").upsert(self._json(row), on_conflict="rollout_id").execute()
        return rid

    def load_qc_rows(self, experiment: str | None = None) -> list[dict]:
        """Read qc_rollouts back into training-ready dicts {..., chunks:[{obs_enc, chunk_pos,
        steps:[{step_idx, s, z_hat}]}]} by downloading each rollout's per-chunk parquet blob."""
        import pandas as pd
        q = self.client.table("qc_rollouts").select("*")
        if experiment:
            q = q.eq("experiment", experiment)
        rows = q.execute().data or []
        out = []
        for r in rows:
            chunks = []
            if r.get("chunks_path"):
                df = pd.read_parquet(io.BytesIO(self._download(r["chunks_path"])))
                for ci, g in df.groupby("chunk_idx"):
                    g0 = g.iloc[0]
                    chunks.append({
                        "chunk_idx": int(ci), "chunk_pos": float(g0["chunk_pos"]),
                        "obs_enc": list(g0["obs_enc"]),
                        "steps": [{"step_idx": int(s["step_idx"]), "s": float(s["s"]),
                                   "z_hat": list(s["z_hat"])} for _, s in g.iterrows()],
                    })
            out.append({"rollout_id": r["rollout_id"], "suite": r["suite"],
                        "task_idx": r["task_idx"], "success": r["success"], "chunks": chunks})
        return out

    def log_qc_eval(self, row: dict) -> None:
        row.setdefault("run_id", self.run_id)
        row.setdefault("experiment", self.experiment)
        self.client.table("qc_eval").upsert(self._json(row),
                                            on_conflict="rollout_id,lambda").execute()

    def register_q_corrector(self, q_ckpt_id: str, ckpt_bytes: bytes, meta: dict,
                             split_ids: dict | None = None) -> str:
        ckpt_key = f"q_correctors/{q_ckpt_id}.pt"
        self._upload(ckpt_key, ckpt_bytes)
        row = {"q_ckpt_id": q_ckpt_id, "run_id": self.run_id, "experiment": self.experiment,
               "ckpt_path": ckpt_key, **meta}
        if split_ids is not None:
            split_key = f"q_correctors/{q_ckpt_id}.split.json"
            self._upload(split_key, json.dumps(split_ids).encode())
            row["split_path"] = split_key
        self.client.table("q_correctors").upsert(self._json(row), on_conflict="q_ckpt_id").execute()
        return q_ckpt_id

    def load_q_corrector(self, q_ckpt_id: str):
        import torch
        row = self.client.table("q_correctors").select("*").eq("q_ckpt_id", q_ckpt_id) \
            .single().execute().data
        blob = self._download(row["ckpt_path"])
        return torch.load(io.BytesIO(blob), map_location="cpu"), row

    # ── encoding cache ─────────────────────────────────────────────────────
    @staticmethod
    def encoding_key(model_revision: str, obs_hash: str, lang_hash: str) -> str:
        return hashlib.sha256(f"{model_revision}|{obs_hash}|{lang_hash}".encode()).hexdigest()

    def get_encoding(self, cache_key: str):
        if cache_key in self._enc_lru:
            self._enc_lru.move_to_end(cache_key)
            return self._enc_lru[cache_key]
        row = self.client.table("encoding_cache").select("blob_path").eq(
            "cache_key", cache_key).execute().data
        if not row:
            return None
        arr = np.load(io.BytesIO(self._download(row[0]["blob_path"])))
        self._enc_put_lru(cache_key, arr)
        return arr

    def put_encoding(self, cache_key: str, arrays: dict, model_revision: str = "",
                     obs_hash: str = "", lang_hash: str = "") -> None:
        key = f"encodings/{cache_key}.npz"
        self._upload(key, _npz_bytes(arrays))
        dims = int(next(iter(arrays.values())).shape[-1]) if arrays else None
        self.client.table("encoding_cache").upsert({
            "cache_key": cache_key, "model_revision": model_revision, "obs_hash": obs_hash,
            "lang_hash": lang_hash, "dims": dims, "blob_path": key}, on_conflict="cache_key").execute()
        self._enc_put_lru(cache_key, np.load(io.BytesIO(_npz_bytes(arrays))))

    def _enc_put_lru(self, key, val):
        self._enc_lru[key] = val
        self._enc_lru.move_to_end(key)
        while len(self._enc_lru) > self._enc_lru_size:
            self._enc_lru.popitem(last=False)

    # ── Storage primitives ─────────────────────────────────────────────────
    def _upload(self, key: str, data: bytes) -> None:
        self.client.storage.from_(self.bucket).upload(
            key, data, {"content-type": "application/octet-stream", "upsert": "true"})
        self._bytes_written += len(data)

    def _download(self, key: str) -> bytes:
        return self.client.storage.from_(self.bucket).download(key)

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    # ── util ───────────────────────────────────────────────────────────────
    @staticmethod
    def _json(row: dict) -> dict:
        return {k: _to_jsonable(v) for k, v in row.items()}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
