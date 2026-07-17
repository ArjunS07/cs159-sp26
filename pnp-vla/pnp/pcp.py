"""PCP: Q-corrector model, training (wandb), collection + correction sampler strategies.

Ported from build_final_notebooks.py PCP sections. The Q-corrector is a small calibrated
MLP success-predictor; PCP nudges actions along its input-gradient at deploy time.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .config import PCPConfig
from .pnp import _pnp_gen


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class QCorrector(nn.Module):
    """Q(z_hat, obs, [chunk_pos, s]) ~= P(success). 3-layer MLP + residual."""

    def __init__(self, action_feat_dim=350, obs_feat_dim=2048, pos_dim=2, hidden=256, dropout=0.2):
        super().__init__()
        self.action_norm = nn.LayerNorm(action_feat_dim)
        self.obs_proj = nn.Linear(obs_feat_dim, 128)
        self.obs_norm = nn.LayerNorm(128)
        self.net = nn.Sequential(
            nn.Linear(action_feat_dim + 128 + pos_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.res = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)

    def forward(self, action, chunk_pos, obs):
        obs_h = self.obs_norm(self.obs_proj(obs))
        h = self.net(torch.cat([self.action_norm(action), obs_h, chunk_pos], dim=-1))
        return self.out(h + self.res(h)).squeeze(-1)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / self.log_t.exp()


# ─────────────────────────────────────────────────────────────────────────────
# Shared K predict-and-perturb -> full-width mean clean estimate (z_hat)
# ─────────────────────────────────────────────────────────────────────────────
def _kpp_zhat(x_t, s, vf, K, adim=7):
    """Run K predict/perturb iters (isolated generator); return (z_hat[chunk,adim], x_acc)."""
    gen = _pnp_gen(x_t.device)
    x_acc = x_t
    a_hats = []
    for _ in range(K):
        v = vf(x_acc)
        a_hat = x_acc - s * v
        a_hats.append(a_hat)
        x_acc = (1.0 - s) * a_hat + s * torch.empty_like(x_acc).normal_(generator=gen)
    z_full = torch.stack(a_hats, 0).mean(0)              # (B, chunk, max_adim)
    z_hat = z_full[0, :, :adim]                          # (chunk, adim)
    return z_hat, x_acc


# ─────────────────────────────────────────────────────────────────────────────
# Sampler strategies (duck-typed for pnp.sampler)
# ─────────────────────────────────────────────────────────────────────────────
class CollectStrategy:
    """Measurement-only: record mean z_hat + obs_enc at collect steps. Non-invasive."""
    invasive = False

    def __init__(self, cfg: PCPConfig, adim=7):
        self.cfg = cfg
        self.adim = adim
        self.chunks = []          # accumulated across the episode: {chunk_idx, obs_enc, chunk_pos, steps}
        self._buf = []

    def selected(self, step, s):
        return step in tuple(self.cfg.collect_steps)

    def step(self, x_t, s, vf, ctx):
        z_hat, _ = _kpp_zhat(x_t, s, vf, self.cfg.pnp_k, self.adim)
        self._buf.append({"step_idx": ctx.step, "s": float(s),
                          "z_hat": z_hat.detach().float().cpu().numpy()})
        return x_t                 # non-invasive: executed action stays vanilla

    def finish(self, ctx):
        self.chunks.append({
            "chunk_idx": len(self.chunks),
            "obs_enc": ctx.obs_enc.detach().float().cpu().numpy(),
            "chunk_pos": ctx.chunk_pos,
            "steps": list(self._buf),
        })
        self._buf = []


class CorrectStrategy:
    """PCP: gated Q-gradient nudge at correction steps. Invasive."""
    invasive = True

    def __init__(self, cfg: PCPConfig, q_model, q_scaler, lam, device, adim=7):
        self.cfg = cfg
        self.q_model = q_model
        self.q_scaler = q_scaler
        self.lam = lam
        self.device = device
        self.adim = adim
        # telemetry
        self.n_applied = 0
        self.n_gate_fire = 0
        self.n_steps = 0
        self.corr_norms = []
        self.q_scores = []

    def selected(self, step, s):
        return step in tuple(self.cfg.correction_steps)

    def step(self, x_t, s, vf, ctx):
        z_hat, x_acc = _kpp_zhat(x_t, s, vf, self.cfg.pnp_k, self.adim)  # z_hat (chunk, adim)
        self.n_steps += 1
        if self.lam and self.lam > 0 and self.q_model is not None:
            zc = z_hat.reshape(1, -1).detach().clone().requires_grad_(True)
            cp = torch.tensor([[float(ctx.chunk_pos), float(s)]], dtype=torch.float32,
                              device=zc.device)
            ob = ctx.obs_enc.reshape(1, -1).float()
            with torch.enable_grad():
                score = torch.sigmoid(self.q_scaler(self.q_model(zc, cp, ob)))
            self.q_scores.append(float(score.item()))
            if score.item() < self.cfg.q_gate:
                self.n_gate_fire += 1
                score.backward()
                g = zc.grad.detach().reshape(z_hat.shape)
                a_star = z_hat + self.lam * g
                self.n_applied += 1
                self.corr_norms.append(float((self.lam * g).norm().item()))
                gen = _pnp_gen(x_t.device)
                # re-noise the corrected full-width estimate (pad correction into first adim)
                z_full = x_acc.clone()
                z_full[0, :, :self.adim] = a_star
                return (1.0 - s) * z_full + s * torch.empty_like(x_t).normal_(generator=gen)
        return x_acc

    def telemetry(self):
        return dict(
            n_corrections_applied=self.n_applied,
            gate_fire_rate=self.n_gate_fire / max(self.n_steps, 1),
            mean_correction_norm=float(np.mean(self.corr_norms)) if self.corr_norms else 0.0,
            mean_q_score=float(np.mean(self.q_scores)) if self.q_scores else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset assembly (hard-task filter)
# ─────────────────────────────────────────────────────────────────────────────
def load_qc_samples(qc_rows, cfg: PCPConfig):
    """qc_rows: iterable of dicts {rollout_id, suite, task_idx, success, chunks:[...]}.

    Returns {(rollout_id, success): [(z_flat, obs_enc, [chunk_pos, s], success), ...]} over the
    hard tasks (SR in (hard_lo, hard_hi)) at the correction steps only.
    """
    task_succ = defaultdict(list)
    for r in qc_rows:
        task_succ[(r["suite"], r["task_idx"])].append(int(r["success"]))
    hard = {k for k, v in task_succ.items() if cfg.hard_lo < (sum(v) / len(v)) < cfg.hard_hi}
    print(f"hard tasks (SR in {cfg.hard_lo}-{cfg.hard_hi}): {len(hard)} / {len(task_succ)}")
    by_rollout = defaultdict(list)
    for r in qc_rows:
        if (r["suite"], r["task_idx"]) not in hard:
            continue
        succ = int(r["success"])
        for c in r["chunks"]:
            for st in c["steps"]:
                if st["step_idx"] not in tuple(cfg.correction_steps):
                    continue
                z = np.asarray(st["z_hat"], dtype=np.float32).reshape(-1)
                by_rollout[(r["rollout_id"], succ)].append(
                    (z, np.asarray(c["obs_enc"], dtype=np.float32),
                     np.asarray([c["chunk_pos"], st["s"]], dtype=np.float32), succ))
    return by_rollout


def _dataset_hash(samples) -> str:
    h = hashlib.sha256()
    for k in sorted(samples, key=lambda x: str(x)):
        h.update(str(k).encode())
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Training (AdamW + cosine + label-smoothed BCE, early-stop on val AUC, temp cal)
# ─────────────────────────────────────────────────────────────────────────────
def train_q_corrector(samples, device, *, seed=42, train_frac=0.80, lr=3e-4, weight_decay=1e-4,
                      label_smooth=0.05, epochs=100, patience=20, batch=256, cfg=None,
                      wandb_run=None, experiment=None):
    """Train + calibrate the Q-corrector. Returns (q_model, q_scaler, meta, split_ids)."""
    import torch.optim as optim
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    except Exception:
        roc_auc_score = average_precision_score = brier_score_loss = None

    cfg = cfg or PCPConfig()
    rng = np.random.default_rng(seed)
    pos = [k for k in samples if k[1] == 1]
    neg = [k for k in samples if k[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    tr_keys = set(pos[:int(len(pos) * train_frac)]) | set(neg[:int(len(neg) * train_frac)])
    split_ids = {"train": [k[0] for k in tr_keys],
                 "val": [k[0] for k in samples if k not in tr_keys]}

    def stack(keys):
        Z, O, P, Y = [], [], [], []
        for k in keys:
            for z, o, p, y in samples[k]:
                Z.append(z); O.append(o); P.append(p); Y.append(y)
        return (torch.tensor(np.array(Z)), torch.tensor(np.array(O)),
                torch.tensor(np.array(P)), torch.tensor(np.array(Y), dtype=torch.float32))

    Ztr, Otr, Ptr, Ytr = stack([k for k in samples if k in tr_keys])
    Zva, Ova, Pva, Yva = stack([k for k in samples if k not in tr_keys])
    action_dim = Ztr.shape[1]
    obs_dim = Otr.shape[1]
    print(f"train chunks={len(Ytr)}  val chunks={len(Yva)}  action_dim={action_dim} obs_dim={obs_dim}")

    q = QCorrector(action_dim, obs_dim).to(device)
    opt = optim.AdamW(q.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 50)
    pos_w = torch.tensor([(Ytr == 0).sum() / max((Ytr == 1).sum(), 1)], device=device)

    def smooth_bce(logits, y):
        y = y * (1 - label_smooth) + 0.5 * label_smooth
        return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        q.train()
        idx = torch.randperm(len(Ytr))
        for i in range(0, len(Ytr), batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            logit = q(Ztr[b].to(device), Ptr[b].to(device), Otr[b].to(device))
            loss = smooth_bce(logit, Ytr[b].to(device))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()
        sched.step()
        q.eval()
        with torch.no_grad():
            vlogit = q(Zva.to(device), Pva.to(device), Ova.to(device)).cpu()
            tlogit = q(Ztr.to(device), Ptr.to(device), Otr.to(device)).cpu()
        vp = torch.sigmoid(vlogit).numpy()
        auc = (roc_auc_score(Yva.numpy(), vp) if roc_auc_score and Yva.unique().numel() > 1
               else float("nan"))
        if wandb_run is not None:
            log = {"train/loss": float(loss), "val/roc_auc": auc, "lr": sched.get_last_lr()[0],
                   "grad_norm": float(grad_norm), "val/best_auc": max(best_auc, auc)}
            if roc_auc_score and Ytr.unique().numel() > 1:
                log["train/roc_auc"] = roc_auc_score(Ytr.numpy(), torch.sigmoid(tlogit).numpy())
            if average_precision_score and Yva.unique().numel() > 1:
                log["val/pr_auc"] = average_precision_score(Yva.numpy(), vp)
            if brier_score_loss:
                log["val/brier"] = brier_score_loss(Yva.numpy(), vp)
            wandb_run.log(log, step=ep)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.cpu().clone() for k, v in q.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            print(f"early stop @ {ep}"); break
    if best_state:
        q.load_state_dict(best_state)

    # temperature calibration
    scaler = TemperatureScaler().to(device)
    copt = optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=100)
    q.eval()
    with torch.no_grad():
        vlogit = q(Zva.to(device), Pva.to(device), Ova.to(device)).detach()
    yv = Yva.to(device)

    def closure():
        copt.zero_grad()
        l = nn.functional.binary_cross_entropy_with_logits(scaler(vlogit), yv)
        l.backward(); return l

    if len(yv):
        copt.step(closure)

    meta = {"action_dim": int(action_dim), "obs_dim": int(obs_dim), "val_auc": float(best_auc),
            "n_pos": len(pos), "n_neg": len(neg), "hard_lo": cfg.hard_lo, "hard_hi": cfg.hard_hi,
            "correction_steps": list(cfg.correction_steps),
            "train_dataset_hash": _dataset_hash(samples), "source_experiment": experiment}
    return q, scaler, meta, split_ids


def ckpt_bytes(q_model, q_scaler, meta) -> bytes:
    buf = io.BytesIO()
    torch.save({"model": q_model.state_dict(), "scaler": q_scaler.state_dict(), **meta}, buf)
    return buf.getvalue()


def new_q_ckpt_id() -> str:
    return uuid.uuid4().hex[:16]
