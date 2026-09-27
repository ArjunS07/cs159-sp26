"""Online SmolVLA Q10 transfer diagnostic with ordinary, unrefined proposals.

The selected actions are P&P-independent.  The currently available Q10 was
trained under a P&P continuation, however, so this is deliberately labelled a
policy-transfer experiment rather than a validated Q^stock estimate.
"""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS, LIBERO_RENDER_LEAD, LIBERO_SKIP_RENDERS,
    _prepare_libero_episodes, _run_collection, identity_shard,
)
from .smolvla_followup_experiments import _matched_historical_stock
from .smolvla_q_selection import SmolSuccessScorer
from .store import SupabaseStore, gather_provenance


EXPERIMENT = "smolvla-libero-q10-vanilla-rerank-transfer-v1"
GUIDANCE_EXPERIMENT = "smolvla-libero-q10-latent-guidance-transfer-v1"


def run_vanilla_rerank_transfer_worker(*, checkpoint_path: str,
                                       shard_index: int = 0, shard_count: int = 2,
                                       episode_limit: int | None = 1,
                                       candidate_count: int = 9,
                                       candidate_generation_batch_size: int = 1,
                                       rollout_batch_size: int = 8):
    """Evaluate Q argmax over ordinary draws, with no P&P in the rollout.

    Episode indices 0--9 are separate from the 10--29 Q training roots.  The
    default is a one-identity smoke test; set episode_limit=None for the shard.
    """
    from . import models

    if shard_count != 2 or shard_index not in range(shard_count):
        raise ValueError("fixed evaluation split is two shards")
    if candidate_count < 2:
        raise ValueError("reranking needs at least two candidates")
    if not 1 <= candidate_generation_batch_size <= candidate_count:
        raise ValueError("invalid candidate generation batch size")
    if episode_limit is not None and episode_limit < 1:
        raise ValueError("episode_limit must be positive or None")
    all_episodes = _prepare_libero_episodes()
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    if episode_limit is not None:
        episodes = episodes[:episode_limit]
    store = SupabaseStore()
    stock = _matched_historical_stock(store, all_episodes)
    scorer = SmolSuccessScorer(checkpoint_path)
    policy, preprocess, postprocess = models.load_smolvla()
    config = RolloutConfig(
        num_samples=candidate_count, candidate_seed_scheme="stock_slot0_v1",
        qplanning_ckpt_id=scorer.checkpoint_id, qplanning_n_elites=1,
        qplanning_temperature=1.0,
        qplanning_candidate_batch_size=candidate_generation_batch_size,
        qplanning_scorer=scorer, num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS, save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS, render_lead=LIBERO_RENDER_LEAD,
    )
    print({"experiment": EXPERIMENT, "shard": f"{shard_index}/{shard_count}",
           "episodes_this_call": len(episodes), "candidates": candidate_count,
           "proposal_distribution": "ordinary SmolVLA draws; no P&P",
           "candidate_generation_batch_size": candidate_generation_batch_size,
           "critic_training_continuation": "P&P; transfer mismatch, evaluate empirically",
           "checkpoint": scorer.checkpoint_id}, flush=True)
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=EXPERIMENT, episodes=episodes,
        methods=[(Method.SMOLVLA_Q10_VANILLA_RERANK_TRANSFER, config)],
        cohort="smolvla_libero_indices_0_9_q_transfer", shard_count=shard_count,
        shard_index=shard_index, benchmark="libero",
        driver="smolvla_q10_vanilla_rerank_transfer",
        run_metadata={"model_repo_id": SMOLVLA_REPO_ID,
                      "training_continuation": "smolvla_pnp_steps123_k311",
                      "evaluation_continuation": "q_reranked_vanilla",
                      "candidate_family": "vanilla", "candidate_count": candidate_count,
                      "candidate_generation_batch_size": candidate_generation_batch_size,
                      "candidate_selector": "argmax_q", "pnp_enabled": False,
                      "n_action_steps": LIBERO_ACTION_STEPS,
                      "checkpoint_id": scorer.checkpoint_id},
        matched_reference_outcomes={"historic stock A10": stock},
        report_every=0, report_every_identities=10,
        rollout_batch_size=rollout_batch_size, provenance=provenance,
        resume_completed_only=True,
    )


def run_q10_latent_guidance_transfer_worker(*, checkpoint_path: str,
                                            shard_index: int = 0,
                                            shard_count: int = 2,
                                            episode_limit: int | None = 1,
                                            euler_step: int = 4,
                                            latent_step_rms: float = 0.05,
                                            rollout_batch_size: int = 8):
    """Test a single Q-gradient correction inside ordinary SmolVLA denoising.

    This is a simulator pilot of a Q-guided corrector, not a trained PCP MLP.
    It uses the existing Q-guidance tap and the P&P-tree Q as a transfer test.
    """
    from . import models

    if shard_count != 2 or shard_index not in range(shard_count):
        raise ValueError("fixed evaluation split is two shards")
    if episode_limit is not None and episode_limit < 1:
        raise ValueError("episode_limit must be positive or None")
    all_episodes = _prepare_libero_episodes()
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    if episode_limit is not None:
        episodes = episodes[:episode_limit]
    store = SupabaseStore()
    stock = _matched_historical_stock(store, all_episodes)
    scorer = SmolSuccessScorer(checkpoint_path)
    policy, preprocess, postprocess = models.load_smolvla()
    config = RolloutConfig(
        q_guidance_ckpt_id=scorer.checkpoint_id,
        q_guidance_step=euler_step, q_guidance_step_size=latent_step_rms,
        q_guidance_scorer=scorer, num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS, save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS, render_lead=LIBERO_RENDER_LEAD,
    )
    print({"experiment": GUIDANCE_EXPERIMENT,
           "shard": f"{shard_index}/{shard_count}",
           "episodes_this_call": len(episodes),
           "correction": "single Q-gradient latent update during ordinary denoising",
           "critic_training_continuation": "P&P; transfer mismatch",
           "euler_step": euler_step, "latent_step_rms": latent_step_rms,
           "checkpoint": scorer.checkpoint_id}, flush=True)
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=GUIDANCE_EXPERIMENT,
        episodes=episodes,
        methods=[(Method.SMOLVLA_Q10_LATENT_GUIDANCE_TRANSFER, config)],
        cohort="smolvla_libero_indices_0_9_q_guidance_transfer",
        shard_count=shard_count, shard_index=shard_index, benchmark="libero",
        driver="smolvla_q10_latent_guidance_transfer",
        run_metadata={"model_repo_id": SMOLVLA_REPO_ID,
                      "training_continuation": "smolvla_pnp_steps123_k311",
                      "evaluation_continuation": "q_guided_vanilla",
                      "pnp_enabled": False, "euler_step": euler_step,
                      "latent_step_rms": latent_step_rms,
                      "n_action_steps": LIBERO_ACTION_STEPS,
                      "checkpoint_id": scorer.checkpoint_id},
        matched_reference_outcomes={"historic stock A10": stock},
        report_every=0, report_every_identities=10,
        rollout_batch_size=rollout_batch_size, provenance=provenance,
        resume_completed_only=True,
    )
