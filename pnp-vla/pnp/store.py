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
        # The package lives inside a larger repository. Limit dirty-state provenance to
        # pnp-vla so changes to legacy files elsewhere in the repository do not taint a run.
        "git_dirty": bool(_git("status", "--porcelain", "--", ".", cwd=here)),
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
            "label": self._auto_label(driver), "notes": notes,
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
    def _auto_experiment(driver: str, config: dict | None = None) -> str:
        return f"{driver}-{_dt.date.today().isoformat()}"

    @staticmethod
    def _auto_label(driver: str) -> str:
        return f"{driver}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ── resume ─────────────────────────────────────────────────────────────
    def existing_keys(self, experiment: str, **eq) -> set[str]:
        q = self.client.table("rollouts").select("rollout_id").eq("experiment", experiment)
        for k, v in eq.items():
            q = q.eq(k, v)
        rows = q.execute().data or []
        return {r["rollout_id"] for r in rows}

    def iter_todo(self, experiment: str, episodes, methods, done: set | None = None):
        """Yield (ep, method, config, rid) for each (episode x method) not already logged.

        Collapses the existing_keys + rollout_id + skip boilerplate so a notebook loop is just
        `for ep, name, cfg, rid in store.iter_todo(...): run + log_result`. `methods` is an iterable
        of (name, config) — a dict is accepted too. Pass a precomputed `done` (from existing_keys)
        once outside the env loop to avoid re-querying per task group. Yields ep-outer so an
        episode's paired methods run back-to-back (friendly to the prefix-encoding cache)."""
        methods = list(methods.items()) if isinstance(methods, dict) else list(methods)
        if done is None:
            done = self.existing_keys(experiment)
        for ep in episodes:
            for name, config in methods:
                rid = self.rollout_id(experiment, ep, name, config)
                if rid not in done:
                    yield ep, name, config, rid

    # ── episode logging ────────────────────────────────────────────────────
    def log_episode(self, rollout_row: dict, euler_steps: list[dict] | None = None,
                    action_vectors: list[dict] | None = None,
                    blobs: dict | None = None) -> str:
        """Upsert one rollout (+ its per-step rows + Storage blobs).

        `blobs` may contain: ahats(dict of arrays), trajectory(dict of arrays), generated_chunks,
        obs_frames(list of arrays), video(bytes), pcp_chunks(pandas.DataFrame).
        Their Storage keys are written back onto the row.
        """
        rid = rollout_row["rollout_id"]
        rollout_row.setdefault("run_id", self.run_id)
        rollout_row.setdefault("experiment", self.experiment)
        rollout_row.setdefault("sampler_algo_version", SAMPLER_ALGO_VERSION)
        rollout_row.setdefault("schema_version", SCHEMA_VERSION)
        for name, key, payload in self._blob_specs(rid, blobs or {}):
            self._upload(key, payload)
            rollout_row[f"{name}_path"] = key

        # Stamp after potentially slow Storage uploads, close to the database write.
        rollout_row["updated_at"] = _now()
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
        if blobs.get("generated_chunks"):
            yield "generated_chunks", f"generated_chunks/{rid}.npz", _npz_bytes(
                blobs["generated_chunks"])
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
        """Denormalized config columns for filtering + the logical-config hash key.

        The flat RolloutConfig fields map 1:1 to these columns, so this is near-identity."""
        on = config.has_probe
        return {
            "method": method,
            "pnp_enabled": on,
            "pnp_step_indices": list(config.pnp_steps) if (on and config.pnp_steps) else None,
            "pnp_k": config.pnp_k if on else None,
            "refine_average": (config.refine_average if config.refine else None),
            "pnp_time_min": config.pnp_time_min if on else None,
            "num_inference_steps": config.num_inference_steps,
            "num_samples": config.num_samples,
        }

    @staticmethod
    def _logical_key(method: str, config) -> dict:
        """The full behavior-defining key hashed into rollout_id / config_hash: method + every
        field that changes the trajectory. Sweeping any of it (e.g. correction_lambda) yields a
        distinct rollout_id even under one method name — no silent collisions."""
        return {"method": method, **config.logical_dict()}

    def rollout_id(self, experiment: str, ep: dict, method: str, config) -> str:
        identity = {k: ep.get(k) for k in
                    ("benchmark", "suite", "task_idx", "episode_idx", "ep_idx", "init_state_hash")}
        return self.make_rollout_id(experiment, identity, self._logical_key(method, config))

    @staticmethod
    def _dim_cols(vec, prefix):
        # NumPy arrays deliberately reject boolean coercion (``vec or []``),
        # because it is ambiguous for vectors with more than one element.
        vec = [] if vec is None else list(vec)
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
        denorm = self._denorm(method, config)          # denormalized filter columns
        logical = self._logical_key(method, config)    # full behavior config -> hash + provenance
        row = {
            "rollout_id": rid,
            "benchmark": ep.get("benchmark"), "suite": ep["suite"], "task_idx": ep["task_idx"],
            "task_desc": ep.get("task_desc"), "episode_idx": ep.get("ep_idx", ep.get("episode_idx")),
            "init_state_hash": ep.get("init_state_hash"), "bddl_sha256": ep.get("bddl_sha256"),
            "suite_family": ep.get("suite_family"), "perturb_axis": ep.get("perturb_axis"),
            "perturb_strength": ep.get("perturb_strength"),
            "distractor_object": ep.get("distractor_object"),
            "max_steps": ep.get("max_steps"), "chunk_size": result.get("chunk_size"),
            "n_chunks": result["n_chunks"], "action_dim": ADIM,
            "episode_seed": result["episode_seed"], "perturb_seed": result.get("perturb_seed"),
            "config_hash": self.config_hash(logical),
            "config_json": logical,
            "success": result["success"], "n_steps": result["n_steps"],
            "elapsed_s": result["elapsed_s"],
            "terminated_reason": result.get("terminated_reason"),
            "status": result["status"], "error_msg": result["error_msg"],
            "nan_action_count": result["nan_action_count"], "n_vf_evals": result["n_vf_evals"],
            "inference_ms_total": result.get("inference_ms_total"),
            "started_at": result.get("started_at"), "finished_at": result.get("finished_at"),
            **denorm, **summary, **result["instability"],
        }
        # Keep stock LIBERO compatible with pre-cohort schemas. PRO manifests carry these keys
        # and require the idempotent migration at the bottom of supabase/schema.sql.
        for cohort_key in ("canonical_member", "expanded_member"):
            if cohort_key in ep:
                row[cohort_key] = ep[cohort_key]
        if config.correction_lambda is not None:
            row.update(correction_lambda=config.correction_lambda, q_gate=config.q_gate,
                       correction_steps=list(config.pnp_steps),
                       q_ckpt_id=config.q_ckpt_id, **(result.get("pcp_telemetry") or {}))
        if result.get("ms_selections"):
            sels = result["ms_selections"]
            row["ms_chosen_idx"] = sels[0]["chosen"]
            row["ms_candidate_u"] = {"chosen": [s["chosen"] for s in sels],
                                     "u": [s["cand_u"] for s in sels]}
        blobs = {}
        if result.get("trajectory"):
            blobs["trajectory"] = result["trajectory"]
        if result.get("generated_chunks"):
            blobs["generated_chunks"] = result["generated_chunks"]
        if result.get("obs_frames"):
            blobs["obs_frames"] = result["obs_frames"]
        if result.get("video"):
            blobs["video"] = result["video"]
        if config.save_ahats and result.get("recorder_episode"):
            ah = {f"c{c['chunk_idx']}_s{st['step']}": st["a_hats"]
                  for c in result["recorder_episode"].get("chunks", [])
                  for st in c.get("steps", []) if "a_hats" in st}
            if ah:
                blobs["ahats"] = ah
        if config.save_pcp_features and result.get("pcp_chunks"):
            blobs["pcp_chunks"] = self._pcp_chunks_df(result["pcp_chunks"])
        return self.log_episode(row, euler_steps=euler, action_vectors=vecs, blobs=blobs or None)

    @staticmethod
    def _pcp_chunks_df(pcp_chunks: list):
        """Flatten tap pcp_chunks into a per-(chunk,step) DataFrame for the parquet blob."""
        import pandas as pd
        rows = []
        for c in pcp_chunks:
            for st in c["steps"]:
                rows.append({"chunk_idx": c["chunk_idx"], "chunk_pos": c["chunk_pos"],
                             "obs_enc": np.asarray(c["obs_enc"]).tolist(),
                             "step_idx": st["step_idx"], "s": st["s"],
                             "z_hat": np.asarray(st["z_hat"]).reshape(-1).tolist()})
        return pd.DataFrame(rows) if rows else None

    # ── PCP training data (read straight from rollouts; no separate qc table) ──
    def load_qc_rows(self, experiment: str | None = None) -> list[dict]:
        """Read every rollout carrying a pcp_chunks blob (the save_pcp_features sink) back into
        training-ready dicts {rollout_id, suite, task_idx, success, chunks:[{obs_enc, chunk_pos,
        steps:[{step_idx, s, z_hat}]}]}. The label is rollouts.success — no qc_rollouts table."""
        import pandas as pd
        q = self.client.table("rollouts").select(
            "rollout_id,suite,task_idx,success,pcp_chunks_path").not_.is_("pcp_chunks_path", "null")
        if experiment:
            q = q.eq("experiment", experiment)
        rows = q.execute().data or []
        out = []
        for r in rows:
            chunks = []
            if r.get("pcp_chunks_path"):
                df = pd.read_parquet(io.BytesIO(self._download(r["pcp_chunks_path"])))
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

    # ── Q-corrector registry ─────────────────────────────────────────────────
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

    # ── clean-chunk verifier registry ────────────────────────────────────────
    def register_verifier(self, verifier_id: str, checkpoint: bytes, config: dict, metrics: dict,
                          split_manifest: dict, *, dataset_hash: str = "") -> str:
        ckpt_path = f"verifiers/{verifier_id}.pt"
        split_path = f"verifiers/{verifier_id}.split.json"
        self._upload(ckpt_path, checkpoint)
        self._upload(split_path, json.dumps(split_manifest, sort_keys=True).encode())
        row = {
            "verifier_id": verifier_id, "experiment": self.experiment, "run_id": self.run_id,
            "model_class": config.get("model_class", "CleanChunkVerifier"),
            "obs_dim": config["obs_dim"], "action_dim": config.get("action_dim", ADIM),
            "horizon": config.get("horizon", 50), "prefix_length": config.get("prefix_length"),
            "checkpoint_path": ckpt_path, "split_path": split_path,
            "config_json": config, "metrics_json": metrics, "dataset_hash": dataset_hash,
        }
        self.client.table("verifier_models").upsert(
            self._json(row), on_conflict="verifier_id").execute()
        return verifier_id

    def register_candidate_group(self, group: dict, candidates: list[dict]) -> str:
        """Idempotently persist one exact/fallback pair after its artifacts are uploaded."""
        group_id = group["candidate_group_id"]
        self.client.table("verifier_candidate_groups").upsert(
            self._json(group), on_conflict="candidate_group_id").execute()
        for source in candidates:
            candidate = dict(source)
            blobs = candidate.pop("blobs", {})
            for name in ("policy_chunk", "env_chunk", "observation"):
                path = f"verifier_candidates/{group_id}/{candidate['candidate_id']}.{name}.npz"
                self._upload(path, _npz_bytes(blobs[name]))
                candidate[f"{name}_path"] = path
            candidate["candidate_group_id"] = group_id
            self.client.table("verifier_candidates").upsert(
                self._json(candidate), on_conflict="candidate_id").execute()
        return group_id

    def load_verifier(self, verifier_id: str):
        import torch
        row = (self.client.table("verifier_models").select("*")
               .eq("verifier_id", verifier_id).single().execute().data)
        checkpoint = torch.load(io.BytesIO(self._download(row["checkpoint_path"])),
                                map_location="cpu", weights_only=False)
        return checkpoint, row

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
        blob = _npz_bytes(arrays)
        key = f"encodings/{cache_key}.npz"
        self._upload(key, blob)
        dims = int(next(iter(arrays.values())).shape[-1]) if arrays else None
        self.client.table("encoding_cache").upsert({
            "cache_key": cache_key, "model_revision": model_revision, "obs_hash": obs_hash,
            "lang_hash": lang_hash, "dims": dims, "blob_path": key}, on_conflict="cache_key").execute()
        self._enc_put_lru(cache_key, np.load(io.BytesIO(blob)))

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
