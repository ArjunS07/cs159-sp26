"""RL Token representation pilot on *contextualized* frozen SmolVLA VLM tokens.

This is the reconstruction-bottleneck stage of the RL Token paper, adapted to
SmolVLA.  It is deliberately separate from qplanning_critic.model.rl_token,
which is merely a Q decoder readout parameter.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .qplanning_critic.model import pool_prefix_tokens


FEATURE_FORMAT = "smolvla_contextual_rl_token_features_v1"
CHECKPOINT_FORMAT = "smolvla_rl_token_reconstruction_v1"


class RLTokenReconstructor(nn.Module):
    """Encode frozen VLM tokens to one state token; reconstruct autoregressively."""

    def __init__(self, input_dim: int, *, width: int = 256, tokens: int = 128,
                 layers: int = 2, heads: int = 8):
        super().__init__()
        if width % heads or tokens < 1:
            raise ValueError("invalid RL Token architecture")
        self.input_dim, self.width, self.tokens = input_dim, width, tokens
        self.layers, self.heads = layers, heads
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.encoder_positions = nn.Parameter(torch.randn(1, tokens + 1, width) * .02)
        self.decoder_positions = nn.Parameter(torch.randn(1, tokens, width) * .02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=.1,
                                         activation="gelu", batch_first=True, norm_first=True)
        dec = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=.1,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, norm=nn.LayerNorm(width))
        self.decoder = nn.TransformerEncoder(dec, layers, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, input_dim)

    def config(self) -> dict:
        return dict(input_dim=self.input_dim, width=self.width, tokens=self.tokens,
                    layers=self.layers, heads=self.heads)

    def encode(self, contextual: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if contextual.ndim != 3 or contextual.shape[1:] != (self.tokens, self.input_dim):
            raise ValueError("contextual tokens have wrong shape")
        if valid.shape != contextual.shape[:2] or not bool(valid.any(1).all()):
            raise ValueError("each example needs at least one valid contextual token")
        h = self.input_projection(contextual.float())
        sequence = torch.cat([h, self.query.expand(len(h), -1, -1)], 1)
        sequence = sequence + self.encoder_positions
        mask = torch.cat([~valid.bool(), torch.zeros((len(h), 1), device=h.device,
                        dtype=torch.bool)], 1)
        return self.encoder(sequence, src_key_padding_mask=mask)[:, -1]

    def forward(self, contextual: torch.Tensor, valid: torch.Tensor):
        """Teacher-forced next-token reconstruction; no future-token leakage."""
        z = self.encode(contextual, valid)
        previous = self.input_projection(contextual[:, :-1].float())
        shifted = torch.cat([z[:, None], previous], 1) + self.decoder_positions
        causal = torch.ones((self.tokens, self.tokens), device=z.device,
                            dtype=torch.bool).triu(1)
        key_padding = torch.cat([torch.zeros((len(z), 1), device=z.device,
                                             dtype=torch.bool), ~valid[:, :-1].bool()], 1)
        predicted = self.output(self.decoder(
            shifted, mask=causal, src_key_padding_mask=key_padding))
        error = (predicted.float() - contextual.float()).square().mean(-1)
        loss = (error * valid.float()).sum() / valid.sum().clamp_min(1)
        return loss, z


@torch.no_grad()
def contextualize_stored_prefix(policy, embeddings, pad, attention, *, tokens=128):
    """Replay the frozen VLM prefill from full stored source-prefix tensors.

    The source artifact retains the full embed_prefix output and its attention
    mask.  Compact v3/v4 branch artifacts retain only pooled prefill embeddings
    and are intentionally rejected here: they cannot reproduce final VLM tokens.
    """
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    model = policy.model
    device = next(model.parameters()).device
    vlm_dtype = next(model.vlm_with_expert.parameters()).dtype
    embedded = torch.as_tensor(embeddings, device=device, dtype=vlm_dtype)
    valid = torch.as_tensor(pad, device=device).bool()
    att = torch.as_tensor(attention, device=device).bool()
    if embedded.ndim == 2:
        embedded = embedded[None]
    if valid.ndim == 1:
        valid = valid[None]
    if att.ndim == 1:
        att = att[None]
    if embedded.ndim != 3 or valid.shape != embedded.shape[:2] or att.shape != valid.shape:
        raise ValueError("full prefix embeddings/masks are required")
    if len(embedded) != 1:
        raise ValueError("extract one decision boundary at a time")
    mask2d = make_att_2d_masks(valid, att)
    position_ids = torch.cumsum(valid, dim=1) - 1
    output, _ = model.vlm_with_expert.forward(
        attention_mask=mask2d, position_ids=position_ids,
        past_key_values=None, inputs_embeds=[embedded, None],
        use_cache=model.config.use_cache, fill_kv_cache=True)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output) or output.ndim != 3 or output.shape[:2] != valid.shape:
        raise RuntimeError("SmolVLA prefill did not return contextual token states; inspect pinned LeRobot API")
    pooled, pooled_valid = pool_prefix_tokens(output, valid, tokens)
    return pooled[0].to("cpu", torch.float16).numpy(), pooled_valid[0].cpu().numpy()


def _feature_name(rollout_id: str) -> str:
    return hashlib.sha256(rollout_id.encode()).hexdigest()[:24] + ".npz"


def prepare_contextual_features(*, output_root: str | Path, train_rollout_ids: set[str],
                                validation_rollout_ids: set[str], limit: int | None = None,
                                store=None, policy=None) -> dict:
    """Cache final VLM tokens by source rollout, preserving the root split.

    `limit=1` is a cheap API/shape smoke test.  Run the full extraction only
    after the source cohort and immutable root split have been verified.
    """
    from .qplanning_fork_pilot import _load_selected_training_arrays
    from .smolvla_tree_collection import _source_rows
    from .store import SupabaseStore

    store = store or SupabaseStore()
    rows = _source_rows(store)
    wanted = train_rollout_ids | validation_rollout_ids
    if train_rollout_ids & validation_rollout_ids or {r["rollout_id"] for r in rows} != wanted:
        raise ValueError("source rollout IDs must partition the exact 800-row cohort")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root = Path(output_root).expanduser()
    selected = rows if limit is None else rows[:limit]
    if policy is None:
        from .models import load_smolvla
        policy, _, _ = load_smolvla()
    policy.eval()
    fields = ("prefix/prefix_embeddings", "prefix/prefix_pad_masks",
              "prefix/prefix_attention_masks")
    counts = {"train": 0, "validation": 0, "reused": 0}
    for index, row in enumerate(selected, 1):
        split = "train" if row["rollout_id"] in train_rollout_ids else "validation"
        path = root / split / _feature_name(row["rollout_id"])
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if str(old["format"]) != FEATURE_FORMAT or str(old["rollout_id"]) != row["rollout_id"]:
                    raise ValueError(f"feature provenance mismatch: {path}")
            counts["reused"] += 1
            continue
        arrays = _load_selected_training_arrays(store, row["training_data_path"], names=fields)
        all_tokens, all_valid = [], []
        for embedded, valid, att in zip(*(arrays[name] for name in fields), strict=True):
            values, mask = contextualize_stored_prefix(policy, embedded, valid, att)
            all_tokens.append(values)
            all_valid.append(mask)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, format=FEATURE_FORMAT,
                            rollout_id=row["rollout_id"],
                            feature_stage="post_vlm_prefill_final_hidden",
                            contextual=np.asarray(all_tokens, np.float16),
                            valid=np.asarray(all_valid, bool))
        os.replace(temp, path)
        counts[split] += 1
        if index % 20 == 0 or index == len(selected):
            print({"source_rollouts_processed": index, "of": len(selected), **counts}, flush=True)
    return counts


class ContextualFeatureDataset(Dataset):
    def __init__(self, directory: str | Path):
        self.files = sorted(Path(directory).glob("*.npz"))
        self.rows = []
        self.cache = OrderedDict()
        for path in self.files:
            with np.load(path, allow_pickle=False) as file:
                if str(file["format"]) != FEATURE_FORMAT or str(file["feature_stage"]) != "post_vlm_prefill_final_hidden":
                    raise ValueError(f"wrong feature stage: {path}")
                self.rows.extend((path, i) for i in range(len(file["contextual"])))
        if not self.rows:
            raise ValueError(f"no contextualized features in {directory}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, local = self.rows[index]
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as file:
                self.cache[path] = (file["contextual"].copy(), file["valid"].copy())
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        values, mask = self.cache[path]
        return torch.from_numpy(values[local].astype(np.float32)), torch.from_numpy(mask[local])


def train_rl_token(*, feature_root: str | Path, output_path: str | Path,
                   epochs: int = 20, batch_size: int = 16, learning_rate: float = 1e-4,
                   device: str | None = None) -> dict:
    """Train only the reconstruction bottleneck; VLA and critic remain frozen."""
    torch.manual_seed(42)
    root = Path(feature_root)
    train = ContextualFeatureDataset(root / "train")
    validation = ContextualFeatureDataset(root / "validation")
    first, _ = train[0]
    model = RLTokenReconstructor(first.shape[-1], tokens=first.shape[0])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, mask in DataLoader(train, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(values.to(device), mask.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        heldout = []
        with torch.no_grad():
            for values, mask in DataLoader(validation, batch_size=batch_size):
                loss, _ = model(values.to(device), mask.to(device))
                heldout.append(float(loss))
        record = {"epoch": epoch, "train_mse": float(np.mean(losses)),
                  "validation_mse": float(np.mean(heldout))}
        history.append(record)
        print(record, flush=True)
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp")
        torch.save({"format": CHECKPOINT_FORMAT, "architecture": model.config(),
                    "feature_stage": "post_vlm_prefill_final_hidden",
                    "feature_root": str(root), "history": history,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, temp)
        os.replace(temp, output)
    return {"checkpoint": str(output), "train_states": len(train),
            "validation_states": len(validation), "history": history}


def materialize_root_tokens(*, snapshot: dict, cache: dict, source_rows: list[dict],
                            checkpoint: str | Path,
                            feature_root: str | Path, output_root: str | Path,
                            device: str | None = None) -> dict:
    """Freeze one RLT state vector per source root for an MC-Q comparison."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("feature_stage") != "post_vlm_prefill_final_hidden":
        raise ValueError("expected contextual RL Token reconstruction checkpoint")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:20]
    model = RLTokenReconstructor(**payload["architecture"])
    model.load_state_dict(payload["model"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval().requires_grad_(False)
    root = Path(output_root).expanduser()
    groups = snapshot["groups"]
    source_by_path = {str(row["training_data_path"]): str(row["rollout_id"])
                      for row in source_rows}
    counts = {"train": 0, "validation": 0}
    if cache["snapshot_digest"] != snapshot["snapshot_digest"]:
        raise ValueError("cache/snapshot mismatch")
    train_ids = set(cache["train_group_ids"])
    val_ids = set(cache["validation_group_ids"])
    if train_ids & val_ids or train_ids | val_ids != {g["candidate_group_id"] for g in groups}:
        raise ValueError("snapshot root split is incomplete")
    with torch.no_grad():
        for group in groups:
            group_id = group["candidate_group_id"]
            split = "train" if group_id in train_ids else "validation"
            stock = group["candidates"][0]
            if stock["candidate_kind"] != "stored_source":
                raise ValueError("RLT root features require a stored-source candidate first")
            rollout_id = source_by_path[str(stock["training_data_path"])]
            boundary = int(stock["training_data_start_boundary"])
            path = Path(feature_root) / split / _feature_name(rollout_id)
            with np.load(path, allow_pickle=False) as file:
                values = torch.from_numpy(file["contextual"][boundary].astype(np.float32))[None].to(device)
                mask = torch.from_numpy(file["valid"][boundary])[None].to(device)
            token = model.encode(values, mask)[0].float().cpu().numpy()
            destination = root / (hashlib.sha256(group_id.encode()).hexdigest()[:24] + ".npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temp, group_id=group_id, token=token,
                                checkpoint_digest=digest, snapshot_digest=snapshot["snapshot_digest"])
            os.replace(temp, destination)
            counts[split] += 1
    manifest = {"format": "smolvla_rl_token_root_features_v1",
                "rlt_checkpoint_digest": digest,
                "snapshot_digest": snapshot["snapshot_digest"], **counts}
    pending = root / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    os.replace(pending, root / "manifest.json")
    return {"root_token_dir": str(root), **manifest}
