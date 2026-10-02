"""Replicate the saved Jeff stock cohort before interpreting PCP versus stock."""
import json
from .config import RolloutConfig
from .experiments import _prepare_libero_episodes, identity_shard, _run_collection
from .store import SupabaseStore, gather_provenance

EXPERIMENT='smolvla-libero-jeff-stock-replication-v2'
REFERENCE='smolvla-libero-a10-pnp-k5-steps34-v1'
METHOD='smolvla_stock_jeff_replication'
REPORT_KEY=f'smolvla_pcp_sweep/{EXPERIMENT}/comparison.json'


def configure_precision():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    # Jeff did not override or record cuDNN TF32. Preserve the pinned Torch default.


def identity(row):
    return (row['suite'],int(row['task_idx']),int(row.get('episode_idx',row.get('ep_idx'))),row['init_state_hash'])


def compare_rows(reference, observed):
    old={identity(r):r for r in reference}
    new={identity(r):r for r in observed if r['status']=='completed'}
    paired=old.keys()&new.keys()
    flips=[dict(suite=k[0],task_idx=k[1],episode_idx=k[2],init_state_hash=k[3],
                historical_success=old[k]['success'],replicated_success=new[k]['success'])
           for k in sorted(paired) if bool(old[k]['success'])!=bool(new[k]['success'])]
    seed_mismatches=sum(old[k]['episode_seed']!=new[k]['episode_seed'] for k in paired)
    step_mismatches=sum(old[k]['n_steps']!=new[k]['n_steps'] for k in paired)
    complete=len(old)==400 and new.keys()==old.keys()
    return dict(reference_experiment=REFERENCE,replication_experiment=EXPERIMENT,
                historical_episodes=len(old),completed=len(new),paired=len(paired),
                historical_successes=sum(r['success'] for r in old.values()),
                replicated_successes=sum(r['success'] for r in new.values()),
                outcome_matches=len(paired)-len(flips),outcome_flips=flips,
                seed_mismatches=seed_mismatches,step_count_mismatches=step_mismatches,
                complete=complete,outcomes_replicated=complete and not flips and not seed_mismatches,
                interpretation='Outcome agreement does not establish bitwise equality of actions or images.')


def comparison(store=None):
    store=store or SupabaseStore()
    fields='suite,task_idx,episode_idx,init_state_hash,status,success,episode_seed,n_steps'
    reference=store.fetch_all('rollouts',fields,configure=lambda q:q.eq('experiment',REFERENCE)
        .eq('method','pnp_uncertainty_only').eq('status','completed'),order_by=('rollout_id',))
    if len(reference)!=400 or sum(r['success'] for r in reference)!=255:
        raise RuntimeError('Historical stock reference is not the verified 255/400 cohort')
    observed=store.fetch_all('rollouts',fields,configure=lambda q:q.eq('experiment',EXPERIMENT)
        .eq('method',METHOD),order_by=('rollout_id',))
    report=compare_rows(reference,observed)
    store._upload(REPORT_KEY,json.dumps(report,indent=2).encode())
    return report


def historical_batch_plan(episodes, reference, runs):
    """Reconstruct batches from saved run order and previously completed stock rows.

    Resume boundaries inside a run and overwritten attempts are not recoverable. Persist
    this reconstruction and its assumptions instead of claiming exact historical parity.
    Companion lanes are retained even if their target belongs to a subsequent source run.
    """
    by_identity = {identity(e): e for e in episodes}
    if set(by_identity) != {identity(r) for r in reference}:
        raise RuntimeError('Simulator starting-state identities differ from Jeff reference')
    prior = set()
    plan = []
    for run in sorted(runs, key=lambda r: (r['created_at'], r['run_id'])):
        targets = {identity(r) for r in reference if r['run_id'] == run['run_id']}
        if not targets:
            continue
        cfg = run['config_json']
        limit = int(cfg['rollout_batch_size'])
        shard = int(cfg['shard_index'])
        source_eps = identity_shard(episodes, int(cfg['shard_count']), shard)
        for task in sorted({(e['suite'], e['task_idx']) for e in source_eps}):
            pending = sorted([e for e in source_eps
                              if (e['suite'], e['task_idx']) == task and identity(e) not in prior],
                             key=lambda e: int(e.get('ep_idx', e.get('episode_idx'))))
            for offset in range(0, len(pending), limit):
                members = pending[offset:offset + limit]
                selected = [identity(e) for e in members if identity(e) in targets]
                if selected:
                    plan.append(dict(source_run=run['run_id'], source_sha=run['pnp_git_sha'],
                                     shard=shard, batch_limit=limit, members=members,
                                     targets=selected,
                                     skip_unused_renders=(len(members) == 1 or run['pnp_git_sha'].startswith('8ad4afe'))))
        prior.update(targets)
    covered = [k for batch in plan for k in batch['targets']]
    if len(covered) != len(reference) or len(set(covered)) != len(reference):
        raise RuntimeError('Historical plan does not cover each reference identity exactly once')
    return plan


def runtime_metadata(policy):
    import hashlib, importlib.metadata as md, platform, subprocess
    from pathlib import Path
    import torch
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'lerobot', 'libero', 'mujoco',
                 'robosuite', 'numpy', 'Pillow', 'safetensors', 'huggingface-hub'):
        try: versions[name] = md.version(name)
        except md.PackageNotFoundError: versions[name] = None
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        source.update(str(path.relative_to(Path(__file__).parent)).encode())
        source.update(path.read_bytes())
    snapshot = Path(policy._pnp_policy_snapshot)
    weights = {}
    for path in sorted(snapshot.rglob('*.safetensors')):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
        weights[str(path.relative_to(snapshot))] = h.hexdigest()
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                          '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): driver = None
    return dict(packages=versions, python=platform.python_version(), driver=driver,
                source_sha256=source.hexdigest(), policy_revision=policy._pnp_policy_revision,
                policy_weights_sha256=weights, vision_snapshot=policy._pnp_vlm_snapshot,
                cudnn_version=torch.backends.cudnn.version(),
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                num_threads=torch.get_num_threads())


def run_replication(shard_index=0):
    import hashlib
    import torch
    from .models import load_smolvla
    from .libero_env import make_env
    from .rollout import run_episode_batch
    configure_precision()
    if torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('Jeff parity runner requires Torch 2.11.0+cu128')
    all_eps = _prepare_libero_episodes()
    store = SupabaseStore()
    reference = store.fetch_all('rollouts',
        'run_id,suite,task_idx,episode_idx,init_state_hash,success',
        configure=lambda q: q.eq('experiment', REFERENCE).eq('method', 'pnp_uncertainty_only')
                            .eq('status', 'completed'), order_by=('rollout_id',))
    if len(reference) != 400 or sum(r['success'] for r in reference) != 255:
        raise RuntimeError('Historical stock reference changed')
    runs = store.client.table('experiment_runs').select(
        'run_id,created_at,pnp_git_sha,config_json').eq('experiment', REFERENCE).execute().data
    plan = [b for b in historical_batch_plan(all_eps, reference, runs) if b['shard'] == shard_index]
    serializable = [{**b, 'members': [identity(e) for e in b['members']]} for b in plan]
    plan_hash = hashlib.sha256(json.dumps(serializable, sort_keys=True).encode()).hexdigest()
    prefix = f'smolvla_pcp_sweep/{EXPERIMENT}/worker_{shard_index}'
    store._upload(prefix + '/historical_batch_plan.json', json.dumps(dict(
        batches=serializable, plan_sha256=plan_hash,
        assumption='Reconstructed from surviving rows and run creation order; overwritten attempts unknown'),
        indent=2).encode())
    policy, pre, post = load_smolvla(device='cuda')
    runtime = runtime_metadata(policy)
    prov = gather_provenance(model_repo_id='HuggingFaceVLA/smolvla_libero',
                             model_revision=runtime['policy_revision'],
                             weights_sha256=hashlib.sha256(json.dumps(runtime['policy_weights_sha256'],
                                                                     sort_keys=True).encode()).hexdigest())
    prov['policy_model'] = 'smolvla'
    store.start_run(driver='smolvla_jeff_historical_batch_replay', benchmark='libero',
                    experiment=EXPERIMENT, provenance=prov,
                    config=dict(shard_index=shard_index, pnp_enabled=False, q_enabled=False,
                                historical_plan_sha256=plan_hash, runtime=runtime))
    done = store.existing_keys(EXPERIMENT, status='completed')
    completed = 0
    envs = []
    env_key = None
    try:
        for batch in plan:
            cfg = RolloutConfig(n_action_steps=10, num_inference_steps=10,
                                skip_unused_renders=batch['skip_unused_renders'], render_lead=2,
                                save_trajectory=True, video='off')
            assert not cfg.has_probe and not cfg.refine
            eps = [{**e, 'behavior_seed_index': 0} for e in batch['members']]
            target_set = set(batch['targets'])
            ids = [store.rollout_id(EXPERIMENT, e, METHOD, cfg) for e in eps]
            if all(rid in done for e, rid in zip(eps, ids) if identity(e) in target_set):
                continue
            key = (batch['source_run'], eps[0]['suite'], eps[0]['task_idx'])
            if key != env_key:
                for env in envs: env.close()
                envs = []
                env_key = key
            while len(envs) < len(eps): envs.append(make_env(eps[0]['bddl_path']))
            results = run_episode_batch(envs[:len(eps)], eps, policy, pre, post, 'cuda', cfg,
                                        parity_trace=True)
            for e, rid, result in zip(eps, ids, results):
                if identity(e) not in target_set or rid in done: continue
                trace = dict(source_run=batch['source_run'], source_sha=batch['source_sha'],
                             companion_identities=[identity(x) for x in eps],
                             original_batch_limit=batch['batch_limit'],
                             skip_unused_renders=cfg.skip_unused_renders,
                             sampler='direct original stock; no diagnostic probes',
                             trace_path='batched' if len(eps) > 1 else 'historical serial path',
                             checkpoints=result.get('parity_trace', []))
                store._upload(prefix + '/diagnostics/' + rid + '.json', json.dumps(trace).encode())
                store.log_result(rid, e, METHOD, cfg, result)
                if result['status'] != 'completed':
                    raise RuntimeError(result.get('error_msg') or 'Replication episode failed')
                done.add(rid)
                completed += 1
            print(dict(worker=shard_index, completed_new=completed, source_run=batch['source_run'],
                       batch_lanes=len(eps)), flush=True)
    except BaseException:
        store.finish_run(status='failed', n_rollouts=completed)
        raise
    finally:
        for env in envs: env.close()
    store.finish_run(n_rollouts=completed)
    return comparison(store)


def require_replication():
    report=comparison()
    if not report['outcomes_replicated']:
        raise RuntimeError('Jeff stock replication has not matched all 400 outcomes. Inspect '+REPORT_KEY+
                           ': '+json.dumps({k:v for k,v in report.items() if k!='outcome_flips'}))
    return report
