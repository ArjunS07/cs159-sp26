"""Two-arm PCP trial using the accepted notebook-127 historical batch plan."""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
import numpy as np
import torch
from .config import RolloutConfig
from .store import SupabaseStore, gather_provenance
from .smolvla_jeff_replication import (REFERENCE, EXPERIMENT as ACCEPTED, METHOD as ACCEPTED_METHOD,
    identity, historical_batch_plan, configure_precision, runtime_metadata)
from .smolvla_pcp_sweep import SweepTap, load_critic

EXPERIMENT = 'smolvla-pcp-stock-matched127-frozenmc-r002-k3-v1'
CRITIC_SHA = '9b8800bc9f27116afcb1812f93f8a9f6767ac012c7cdac790502032fbb7d0c8e'
ARM = dict(critic='frozen_mc', step=3, radius=.02, every=1, gate=1.01,
           iterations=3, mode='ascent')
METHODS = ('stock', 'pcp')


def arm_config(method, batch):
    cfg = RolloutConfig(n_action_steps=10, num_inference_steps=10,
                        skip_unused_renders=batch['skip_unused_renders'], render_lead=2,
                        save_trajectory=True, video='off')
    if method == 'stock': return cfg
    if method != 'pcp': raise ValueError(method)
    return replace(cfg, pnp_steps=(1,2,3), pnp_k=3, pnp_k_by_step=(3,1,1), refine=True)


def paired_summary(stock, pcp):
    a={identity(r):r for r in stock if r['status']=='completed'}
    b={identity(r):r for r in pcp if r['status']=='completed'}
    keys=sorted(a.keys() & b.keys()); n=len(keys)
    delta=np.array([float(b[k]['success'])-float(a[k]['success']) for k in keys])
    groups=defaultdict(list)
    for k,d in zip(keys,delta): groups[k[:2]].append(float(d))
    task=np.array([np.mean(v) for v in groups.values()])
    se=float(delta.std(ddof=1)/math.sqrt(n)) if n>1 else None
    tse=float(task.std(ddof=1)/math.sqrt(len(task))) if len(task)>1 else None
    failures=sum(not a[k]['success'] for k in keys); successes=n-failures
    rescues=sum(not a[k]['success'] and b[k]['success'] for k in keys)
    spoils=sum(a[k]['success'] and not b[k]['success'] for k in keys)
    return dict(stock_completed=len(a),pcp_completed=len(b),planned_per_arm=400,paired=n,
                stock_successes=sum(a[k]['success'] for k in keys),pcp_successes=sum(b[k]['success'] for k in keys),
                rescues=rescues,spoils=spoils,rescue_rate=rescues/failures if failures else None,
                spoil_rate=spoils/successes if successes else None,
                paired_difference=float(delta.mean()) if n else None,episode_se=se,
                task_groups=len(task),equal_task_weight_difference=float(task.mean()) if len(task) else None,
                equal_task_weight_se=tse,seed_mismatches=sum(a[k]['episode_seed']!=b[k]['episode_seed'] for k in keys),
                complete=len(a)==len(b)==n==400,
                provisional=n<400,interpretation='Exploratory fixed cohort; settings informed by earlier results. No isolated claim about Q versus P&P.')


def comparison(store=None):
    store=store or SupabaseStore()
    fields='suite,task_idx,episode_idx,init_state_hash,status,success,episode_seed,n_steps'
    rows=store.fetch_all('rollouts',fields+',method',configure=lambda q:q.eq('experiment',EXPERIMENT),order_by=('rollout_id',))
    report=paired_summary([r for r in rows if r['method']=='stock'],[r for r in rows if r['method']=='pcp'])
    accepted=store.fetch_all('rollouts',fields,configure=lambda q:q.eq('experiment',ACCEPTED).eq('method',ACCEPTED_METHOD).eq('status','completed'),order_by=('rollout_id',))
    prior={identity(r):r for r in accepted}
    fresh={identity(r):r for r in rows if r['method']=='stock' and r['status']=='completed'}
    shared=prior.keys() & fresh.keys()
    report['stock_vs_127']=dict(paired=len(shared),outcome_matches=sum(prior[k]['success']==fresh[k]['success'] for k in shared),
                               seed_mismatches=sum(prior[k]['episode_seed']!=fresh[k]['episode_seed'] for k in shared))
    store._upload(f'smolvla_pcp_sweep/{EXPERIMENT}/comparison.json',json.dumps(report,indent=2).encode())
    return report


def run_trial(checkpoint_path, shard_index=0):
    if shard_index not in (0,1): raise ValueError('Two disjoint historical shards only')
    if hashlib.sha256(open(checkpoint_path,'rb').read()).hexdigest()!=CRITIC_SHA:
        raise RuntimeError('Frozen MC checkpoint differs from the fixed trial protocol')
    from .experiments import _prepare_libero_episodes
    from .models import load_smolvla
    from .libero_env import make_env
    from .rollout import run_episode_batch
    configure_precision()
    if torch.__version__!='2.11.0+cu128':raise RuntimeError('Use notebook 127 Torch/CUDA versions')
    store=SupabaseStore(); all_eps=_prepare_libero_episodes()
    accepted=store.fetch_all('rollouts','suite,task_idx,episode_idx,init_state_hash,episode_seed,success',
        configure=lambda q:q.eq('experiment',ACCEPTED).eq('method',ACCEPTED_METHOD).eq('status','completed'),order_by=('rollout_id',))
    if len(accepted)!=400 or len({identity(r) for r in accepted})!=400:
        raise RuntimeError('Accepted notebook 127 replication must contain all 400 identities')
    if {identity(e) for e in all_eps}!={identity(r) for r in accepted}:
        raise RuntimeError('Starting-state cohort differs from accepted notebook 127')
    reference=store.fetch_all('rollouts','run_id,suite,task_idx,episode_idx,init_state_hash,success',
        configure=lambda q:q.eq('experiment',REFERENCE).eq('method','pnp_uncertainty_only').eq('status','completed'),order_by=('rollout_id',))
    runs=store.fetch_all('experiment_runs','run_id,created_at,pnp_git_sha,config_json',configure=lambda q:q.eq('experiment',REFERENCE),order_by=('created_at','run_id'))
    plan=[b for b in historical_batch_plan(all_eps,reference,runs) if b['shard']==shard_index]
    serial=[{**b,'members':[identity(e) for e in b['members']]} for b in plan]
    prefix=f'smolvla_pcp_sweep/{EXPERIMENT}/worker_{shard_index}'
    manifest=dict(arm=ARM,critic_sha256=CRITIC_SHA,batches=serial,methods=METHODS,
                  generation_seed_stream=0,planned_per_arm=400,shard=shard_index,
                  scope='Repeated full-episode stock versus P&P plus normalized Q correction; no reranking or sweep')
    manifest_bytes=json.dumps(manifest,sort_keys=True).encode()
    key=prefix+'/protocol.json'
    try: old=store._download(key)
    except Exception as error:
        if not any(x in str(error).lower() for x in ('404','not found','does not exist')):raise
        store._upload(key,manifest_bytes)
    else:
        if old!=manifest_bytes:raise RuntimeError('Saved fixed trial protocol differs')
    policy,pre,post=load_smolvla(device='cuda')
    critic,sha=load_critic(checkpoint_path,'cuda')
    runtime=runtime_metadata(policy)
    prov=gather_provenance(model_repo_id='HuggingFaceVLA/smolvla_libero',model_revision=runtime['policy_revision'])
    prov['policy_model']='smolvla'
    store.start_run('smolvla_pcp_matched127','libero',EXPERIMENT,config={**manifest,'runtime':runtime},provenance=prov)
    done=store.existing_keys(EXPERIMENT,status='completed');completed=0;envs=[];env_key=None
    try:
        for method in METHODS:
            for batch in plan:
                cfg=arm_config(method,batch)
                eps=[{**e,'behavior_seed_index':0} for e in batch['members']]
                targets=set(batch['targets']);ids=[store.rollout_id(EXPERIMENT,e,method,cfg) for e in eps]
                if all(rid in done for e,rid in zip(eps,ids) if identity(e) in targets):continue
                current_key=(method,batch['source_run'],eps[0]['suite'],eps[0]['task_idx'])
                if current_key!=env_key:
                    for env in envs:env.close()
                    envs=[];env_key=current_key
                while len(envs)<len(eps):envs.append(make_env(eps[0]['bddl_path']))
                telemetry=[[] for _ in eps]
                factory=None if method=='stock' else lambda **kw:SweepTap(arm=ARM,critic=critic,correction_std=critic.action_std,
                    emit=lambda lane,item:telemetry[lane].append(item),**kw)
                results=run_episode_batch(envs[:len(eps)],eps,policy,pre,post,'cuda',cfg,tap_factory=factory,parity_trace=True)
                for e,rid,result,chunks in zip(eps,ids,results,telemetry):
                    if identity(e) not in targets or rid in done:continue
                    trace=dict(source_run=batch['source_run'],source_sha=batch['source_sha'],method=method,
                               companions=[identity(x) for x in eps],parity=result.get('parity_trace',[]),corrections=chunks)
                    store._upload(prefix+'/diagnostics/'+rid+'.json',json.dumps(trace,allow_nan=False).encode())
                    result['q_guidance_telemetry']=dict(arm=ARM if method=='pcp' else None,critic_sha=sha if method=='pcp' else None,
                        shard_index=shard_index,diagnostic_path=prefix+'/diagnostics/'+rid+'.json',
                        corrections=sum(x['standardized_rms']>0 for x in chunks),bad_gradients=sum(x['bad_gradient'] for x in chunks))
                    store.log_result(rid,e,method,cfg,result,persist_probe_rows=False)
                    if result['status']!='completed':raise RuntimeError(result.get('error_msg') or 'Episode failed')
                    done.add(rid);completed+=1
                print(json.dumps(dict(worker=shard_index,method=method,completed_new=completed,batch_lanes=len(eps))),flush=True)
                if completed%25==0:print(json.dumps(comparison(store)),flush=True)
    except BaseException:
        store.finish_run(status='failed',n_rollouts=completed);raise
    finally:
        for env in envs:env.close()
    store.finish_run(n_rollouts=completed)
    return comparison(store)
