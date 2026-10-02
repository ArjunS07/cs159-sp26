"""Interpretable recorded-policy Bellman diagnostics, independent of training loss."""
import numpy as np


def recorded_policy_residual(q, rewards, discounts, next_q):
    """For a recorded n-step window, return Q - (accumulated reward + discount Q').

    Caller accumulates rewards/discounts over the specified horizon and selects
    recorded next actions. Horizon termination is encoded in discount=0.
    """
    q, rewards, discounts, next_q = (np.asarray(v, dtype=float) for v in (q, rewards, discounts, next_q))
    if not (q.shape == rewards.shape == discounts.shape == next_q.shape) or q.ndim != 1 or not len(q):
        raise ValueError("Require aligned nonempty vectors")
    if not all(np.isfinite(v).all() for v in (q, rewards, discounts, next_q)):
        raise ValueError("Nonfinite predictions or transition values")
    if any(np.any((v < 0) | (v > 1)) for v in (q, rewards, discounts, next_q)):
        raise ValueError("Success probabilities, accumulated rewards and discounts must lie in [0,1]")
    target = rewards+discounts*next_q
    if np.any(target > 1+1e-7):
        raise ValueError("Target exceeds success-probability range; check terminal masks")
    return q-target


def bellman_summary(q, rewards, discounts, next_q, episode_ids):
    residual = recorded_policy_residual(q,rewards,discounts,next_q)
    ids = np.asarray(episode_ids)
    if ids.shape != residual.shape:
        raise ValueError("Episode IDs must align with transitions")
    groups = [residual[ids == identity] for identity in np.unique(ids)]
    terminal = np.asarray(discounts) == 0
    def stats(values):
        return {"mean":float(values.mean()), "mae":float(np.abs(values).mean()),
                "rmse":float(np.sqrt(np.square(values).mean()))} if len(values) else None
    return {"episodes":len(groups),"transitions":len(residual),
            "episode_weighted":{"mean":float(np.mean([g.mean() for g in groups])),
                                "mae":float(np.mean([np.abs(g).mean() for g in groups])),
                                "rmse":float(np.sqrt(np.mean([np.square(g).mean() for g in groups])))},
            "transition_weighted":stats(residual),"terminal":stats(residual[terminal]),
            "nonterminal":stats(residual[~terminal]),
            "residual_quantiles":np.quantile(residual,[0,.05,.25,.5,.75,.95,1]).tolist()}
