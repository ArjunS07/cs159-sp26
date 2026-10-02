"""Replicate the saved Jeff stock cohort before interpreting PCP versus stock."""
import json
from .config import RolloutConfig
from .experiments import _prepare_libero_episodes, identity_shard, _run_collection
from .store import SupabaseStore, gather_provenance

EXPERIMENT='smolvla-libero-jeff-stock-replication-v1'
REFERENCE='smolvla-libero-a10-pnp-k5-steps34-v1'
METHOD='smolvla_stock_jeff_replication'
REPORT_KEY=f'smolvla_pcp_sweep/{EXPERIMENT}/comparison.json'


def configure_precision():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


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


def run_replication(shard_index=0):
    import torch
    from .models import load_smolvla
    configure_precision()
    all_eps=_prepare_libero_episodes()
    store=SupabaseStore()
    reference=store.fetch_all('rollouts','suite,task_idx,episode_idx,init_state_hash',
        configure=lambda q:q.eq('experiment',REFERENCE).eq('method','pnp_uncertainty_only')
        .eq('status','completed'),order_by=('rollout_id',))
    if {identity(e) for e in all_eps}!={identity(e) for e in reference}:
        raise RuntimeError('Simulator starting-state hashes differ from Jeff reference')
    episodes=identity_shard(all_eps,2,shard_index)
    for e in episodes:e['behavior_seed_index']=0
    cfg=RolloutConfig(n_action_steps=10,num_inference_steps=10,skip_unused_renders=True,
                      render_lead=2,save_trajectory=True,video='off')
    if cfg.has_probe or cfg.refine:raise RuntimeError('Stock replication must have no P&P')
    policy,pre,post=load_smolvla(device='cuda')
    prov=gather_provenance(model_repo_id='HuggingFaceVLA/smolvla_libero')
    prov['policy_model']='smolvla'
    _run_collection(store=store,policy=policy,preprocess=pre,postprocess=post,device='cuda',
        experiment=EXPERIMENT,episodes=episodes,methods=[(METHOD,cfg)],
        cohort='Jeff stock replication: exact state hashes, historical seed stream0',
        shard_count=2,shard_index=shard_index,rollout_batch_size=8,report_every=25,
        resume_completed_only=True,provenance=prov,run_metadata={
            'reference_experiment':REFERENCE,'reference_successes':255,'behavior_seed_index':0,
            'pnp_enabled':False,'q_enabled':False,'tf32':False,'matmul_precision':'highest',
            'policy_snapshot':str(getattr(policy,'name_or_path',''))})
    return comparison(store)


def require_replication():
    report=comparison()
    if not report['outcomes_replicated']:
        raise RuntimeError('Jeff stock replication has not matched all 400 outcomes. Inspect '+REPORT_KEY+
                           ': '+json.dumps({k:v for k,v in report.items() if k!='outcome_flips'}))
    return report
