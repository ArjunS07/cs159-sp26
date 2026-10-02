"""Two-worker, resumable, batched full-episode PCP search. No time cutoff."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

from .config import RolloutConfig, PERTURB_SEED_MASK
from .pnp import run_probe
from .tap import BatchedRolloutTap
from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.model import pool_prefix_tokens
from .smolvla_scalar_returns import ScalarCritic, LateFusionScalarCritic
from .smolvla_anchored_critic import AnchoredCritic

EXPERIMENT = 'smolvla-pcp-a100-jeff-matched-stock-v5'
RADII = (.02, .06)
RUNGS = (400, 2000, 6000)
OBS_KEYS = ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
            'robot0_eef_quat', 'robot0_gripper_qpos')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sweep_grid():
    # Fix sampler timing and intervention frequency; spend replication on a small grid.
    arms = []
    def add(critic, radius, mode='ascent', gate=1.01, iterations=1):
        arms.append(dict(critic=critic, step=3, radius=radius, every=1,
                         gate=gate, iterations=iterations, mode=mode))
    for critic, radius in itertools.product(('frozen_mc', 'cnn_mc', 'td'), RADII):
        add(critic, radius)
        add(critic, radius, mode='descent')
        add(critic, radius, gate=.5)
    for radius in RADII:
        add('none', radius, mode='random')
    # Extra inner-search depth on the two critics with better directional evidence.
    for critic, radius in itertools.product(('frozen_mc', 'td'), RADII):
        add(critic, radius, iterations=3)
    arms.sort(key=lambda a:digest({'search_order':73026, **a}))
    for ordinal, arm in enumerate(arms):
        arm.update(ordinal=ordinal, arm_id=digest(arm)[:16])
    return arms


def baseline_arm():
    return dict(critic='none', step=3, radius=0., every=1, gate=1.01,
                iterations=1, mode='zero', ordinal=-1, arm_id='zero')


def rollout_arm_setup(arm, treatment_config, treatment_factory):
    """The zero control bypasses both P&P and the critic, using stock flow."""
    if arm['mode'] == 'zero':
        from dataclasses import replace
        return replace(treatment_config, pnp_steps=None, pnp_k_by_step=None,
                       pnp_time_min=None, refine=False), None
    return treatment_config, treatment_factory


def load_critic(path, device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    def scalar(architecture):
        a=dict(architecture);family=a.pop('model_family','scalar_decoder')
        dims={k:a.pop(k) for k in ('prefix_dim','robot_dim','proprio_dim')}
        cls=LateFusionScalarCritic if family=='late_fusion_scalar' else ScalarCritic
        if family not in ('scalar_decoder','late_fusion_scalar'):raise ValueError(family)
        return cls(**dims,config=QPlanningModelConfig(**a))
    a=saved['architecture'];family=a.get('model_family','scalar_decoder')
    if family.startswith('anchored'):
        model=AnchoredCritic(scalar(a['base_architecture']),
                            'temporal_cnn' if 'cnn' in family else 'transformer',
                            width=a['width'],delta_scale=a['delta_scale'])
    else:model=scalar(a)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval().requires_grad_(False),hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_move(current, anchor, gradient, valid, std, radius, fraction=1.):
    """Normalized ascent step projected into one total standardized-RMS ball."""
    mask=valid[...,None]
    direction=gradient*std*mask
    denominator=(valid.sum(1)*current.shape[-1]).clamp_min(1).to(current.dtype)
    rms=(direction.square().sum((1,2))/denominator).sqrt()
    usable=torch.isfinite(direction).all((1,2)) & (rms>1e-12)
    direction=torch.nan_to_num(direction,nan=0.,posinf=0.,neginf=0.)/torch.nan_to_num(rms,nan=0.,posinf=0.,neginf=0.).clamp_min(1e-12)[:,None,None]
    delta=(current-anchor)/std+radius*fraction*direction*usable[:,None,None]
    delta*=mask
    size=(delta.square().sum((1,2))/denominator).sqrt()
    delta*=torch.minimum(torch.ones_like(size),radius/size.clamp_min(1e-12))[:,None,None]
    return anchor+delta*std,usable


class SweepTap(BatchedRolloutTap):
    def __init__(self, *, arm, critic, correction_std, emit, episodes, observations, steps,
                 chunk_indices, lane_ids, **kwargs):
        super().__init__(**kwargs)
        from .libero_env import obs_to_policy
        from .rollout import _raw_robot_state
        self.arm,self.critic,self.emit=arm,critic,emit
        self.correction_std=correction_std
        self.chunk_indices,self.lane_ids=chunk_indices,lane_ids
        self.remaining=[ep['max_steps']-step for ep,step in zip(episodes,steps)]
        self.robot=torch.as_tensor(np.stack([np.r_[_raw_robot_state(obs),rem/ep['max_steps']]
            for obs,rem,ep in zip(observations,self.remaining,episodes)]),device=kwargs['device'],dtype=torch.float32)
        self.proprio=torch.stack([obs_to_policy(obs,ep['task_desc'])['observation.state']
            for obs,ep in zip(observations,episodes)]).to(kwargs['device'])

    def step(self,x_t,s,vf,ctx):
        arm=self.arm
        probe=run_probe(x_t,s,vf,k=self.config.probe_k(ctx.step),adim=self.adim,
                        generators=self.generators,record_telemetry=False)
        if self._pending_variable_probe is not None:raise RuntimeError('pending probe')
        self._pending_variable_probe=(probe,int(ctx.step))
        if arm['mode']=='zero' or ctx.step!=arm['step']:
            return probe.x_acc
        full=probe.z_hat_full.detach();anchor=full[:,:10,:7].clone()
        valid=torch.arange(10,device=full.device)[None]<torch.tensor(self.remaining,device=full.device)[:,None]
        selected=torch.tensor([c%arm['every']==0 for c in self.chunk_indices],device=full.device)
        std=self.correction_std
        if self.critic is not None:
            prefix,pad=pool_prefix_tokens(ctx.prefix_embeddings.detach(),ctx.prefix_pad_masks.detach(),128)
        if isinstance(self.critic,AnchoredCritic):
            features,base=self.critic.reference_features(prefix,pad,self.robot,self.proprio,anchor,valid)
            score=lambda a:self.critic.logits_from_features(features,base,a,valid,anchor).reshape(-1).sigmoid()
        elif self.critic is not None:
            score=lambda a:self.critic(prefix,pad,self.robot,self.proprio,a,valid).reshape(-1).sigmoid()
        else:score=None
        candidate=anchor.clone();before=None;after=None;bad=torch.zeros_like(selected)
        if score is not None:
            with torch.enable_grad():
                candidate=anchor.clone().requires_grad_(True)
                before=score(candidate)
                bad |= ~torch.isfinite(before.detach())
                selected &= torch.isfinite(before.detach()) & (before.detach()<arm['gate'])
                gradient,=torch.autograd.grad(before.sum(),candidate)
            for iteration in range(arm['iterations']):
                if iteration:
                    with torch.enable_grad():
                        candidate=candidate.detach().requires_grad_(True)
                        q=score(candidate);gradient,=torch.autograd.grad(q.sum(),candidate)
                gradient=gradient.detach()*(1 if arm['mode']=='ascent' else -1)
                candidate,usable=bounded_move(candidate.detach(),anchor,gradient,valid,std,
                                              arm['radius'],1/arm['iterations'])
                bad |= selected & ~usable
                candidate=torch.where(selected[:,None,None],candidate,anchor)
            with torch.no_grad():after=score(candidate)
        else:
            random=torch.empty_like(anchor)
            for lane,chunk in enumerate(self.chunk_indices):
                seed=int(self.generators[lane].initial_seed()) ^ PERTURB_SEED_MASK ^ ((int(chunk)+1)*1000003)
                gen=torch.Generator(device=full.device).manual_seed(seed%(2**63-1))
                random[lane].normal_(generator=gen)
            # Divide by std because bounded_move forms standardized gradients.
            candidate,_=bounded_move(anchor,anchor,random/std,valid,std,arm['radius'])
            candidate=torch.where(selected[:,None,None],candidate,anchor)
        delta=candidate.detach()-anchor
        if not torch.isfinite(delta).all():raise RuntimeError('nonfinite correction')
        update=torch.zeros_like(full);update[:,:10,:7]=delta
        denominator=(valid.sum(1)*7).clamp_min(1)
        native_rms=(delta.square().sum((1,2))/denominator).sqrt()
        standard_rms=((delta/std).square().sum((1,2))/denominator).sqrt()
        cpu=torch.stack([native_rms,standard_rms,selected.float(),bad.float()],1).detach().cpu().numpy()
        qb=before.detach().cpu().numpy() if before is not None else [None]*len(cpu)
        qa=after.detach().cpu().numpy() if after is not None else [None]*len(cpu)
        for lane,values in enumerate(cpu):
            clean=lambda x:float(x) if x is not None and math.isfinite(float(x)) else None
            self.emit(self.lane_ids[lane],dict(chunk_index=self.chunk_indices[lane],euler_step=int(ctx.step),
                q_before=clean(qb[lane]),q_after=clean(qa[lane]),native_rms=float(values[0]),
                standardized_rms=float(values[1]),gate_open=bool(values[2]),bad_gradient=bool(values[3])))
        return probe.x_acc+(1-float(s))*update

    def after_selected_vfield(self,x_t,s,velocity,ctx):
        # Do not copy unused uncertainty traces to CPU; this changes no sampler arithmetic.
        if self._pending_variable_probe is not None:
            if self._pending_variable_probe[1]!=int(ctx.step):raise RuntimeError('probe step mismatch')
            self._pending_variable_probe=None



def _env_server(connection):
    # Each process owns its EGL contexts; concurrent parent calls never move them across threads.
    from .libero_env import make_env,set_camera_observables
    torch.set_num_threads(1)
    envs=OrderedDict()
    try:
        while True:
            request=connection.recv()
            if request is None:break
            lane,path,operation,args,cameras=request
            key=lane
            try:
                # One environment per active lane. Close its old task BEFORE creating
                # a replacement; retaining task-by-lane caches exhausted Colab host RAM.
                if key in envs and envs[key][0]!=path:
                    envs.pop(key)[1].close()
                if key not in envs:envs[key]=(path,make_env(path))
                env=envs[key][1];envs.move_to_end(key)
                set_camera_observables(env,cameras)
                value=getattr(env,operation)(*args)
                success=bool(env.check_success())
                def prune(obs):
                    return {k:obs[k] for k in OBS_KEYS if k in obs and
                            (cameras or k not in OBS_KEYS[:2])}
                if operation=='step':value=(prune(value[0]),*value[1:])
                elif operation in ('reset','set_init_state'):value=prune(value)
                connection.send((True,value,success))
            except Exception as error:connection.send((False,f'{type(error).__name__}: {error}',False))
    finally:
        for _,env in envs.values():env.close()
        connection.close()


class EnvActor:
    def __init__(self):
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe()
        self.process=ctx.Process(target=_env_server,args=(child,),daemon=True)
        self.process.start();child.close();self.lock=threading.Lock()
    def call(self,lane,path,operation,args,cameras):
        with self.lock:
            try:
                if not self.process.is_alive():raise EOFError('actor already exited')
                self.pipe.send((lane,path,operation,args,cameras))
                if not self.pipe.poll(240):raise RuntimeError('simulator actor stalled')
                ok,value,success=self.pipe.recv()
            except (EOFError,BrokenPipeError,ConnectionResetError) as error:
                self.process.join(.2)
                raise RuntimeError(f'Simulator process exited (exit code {self.process.exitcode}); -9 suggests host RAM exhaustion, -11 a native simulator/GL crash. '+str(error)) from error
            if not ok:raise RuntimeError(value)
            return value,success
    def close(self):
        try:self.pipe.send(None)
        except (EOFError,BrokenPipeError):pass
        self.process.join(5)
        if self.process.is_alive():self.process.terminate();self.process.join(5)
        self.pipe.close()


class EnvProxy:
    def __init__(self,actor,lane,path):self.actor,self.lane,self.path=actor,lane,path;self.cameras=True;self.success=False;self.obs={}
    def _pnp_set_camera_observables(self,enabled):self.cameras=bool(enabled);return True
    def _call(self,operation,*args):
        value,self.success=self.actor.call(self.lane,self.path,operation,args,self.cameras)
        if operation=='step':self.obs.update(value[0]);return (dict(self.obs),*value[1:])
        self.obs.update(value);return dict(self.obs)
    def reset(self):self.obs={};return self._call('reset')
    def set_init_state(self,state):return self._call('set_init_state',state)
    def step(self,action):return self._call('step',action)
    def check_success(self):return self.success
    def close(self):pass


def prepare_cases():
    from .libero_env import init_libero_benchmark,build_final_episodes
    from .experiments import LIBERO_SUITES
    bd=init_libero_benchmark()
    tasks=[(suite,t) for t in range(10) for suite in LIBERO_SUITES]
    # Cover up to all 50 released starting states per task before seed replication.
    eps=build_final_episodes(bd,episode_idxs=list(range(50)),tasks=tasks)
    eps.sort(key=lambda e:(e['ep_idx'],e['task_idx'],e['suite']))
    return [{**ep,'behavior_seed_index':stream} for stream in range(3) for ep in eps]


def clustered_ratio(numerators, denominators, clusters):
    """Cluster sandwich SE for a mean/conditional rate; repeated seeds share a state."""
    n=float(sum(denominators))
    if n==0:return None,None,0
    estimate=float(sum(numerators))/n
    scores={}
    for value,weight,cluster in zip(numerators,denominators,clusters):
        scores[cluster]=scores.get(cluster,0.)+value-estimate*weight
    g=len(scores)
    se=math.sqrt(g/(g-1)*sum(x*x for x in scores.values()))/n if g>1 else None
    return estimate,se,g


def leaderboard(arms,records,case_count,cases=None):
    records=list(records.values()) if isinstance(records,dict) else records
    groups={}
    for record in records:groups.setdefault(record['arm_id'],{})[record['case_index']]=record
    baseline={r['case_index']:r for r in records if r['arm_id']=='zero' and r['status']=='completed'}
    rows=[]
    for arm in arms:
        found={i:r for i,r in groups.get(arm['arm_id'],{}).items() if i<case_count}
        good=[r for r in found.values() if r['status']=='completed']
        paired=[r for r in good if r['case_index'] in baseline]
        rescues=sum(r['success'] and not baseline[r['case_index']]['success'] for r in paired)
        spoils=sum(not r['success'] and baseline[r['case_index']]['success'] for r in paired)
        def cluster(record):
            if cases is None:return record['case_index']
            ep=cases[record['case_index']]
            return (ep['suite'],ep['task_idx'],ep['init_state_hash'])
        success_rate,success_se,n_states=clustered_ratio(
            [int(r['success']) for r in good],[1]*len(good),[cluster(r) for r in good])
        deltas=[int(r['success'])-int(baseline[r['case_index']]['success']) for r in paired]
        paired_change,paired_se,n_paired_states=clustered_ratio(deltas,[1]*len(paired),[cluster(r) for r in paired])
        zero_fail=[int(not baseline[r['case_index']]['success']) for r in paired]
        zero_success=[1-x for x in zero_fail]
        rescue_rate,rescue_se,_=clustered_ratio([int(d==1) for d in deltas],zero_fail,[cluster(r) for r in paired])
        spoil_rate,spoil_se,_=clustered_ratio([int(d==-1) for d in deltas],zero_success,[cluster(r) for r in paired])
        rows.append({**arm,'completed':len(good),'errors':len(found)-len(good),
                     'successes':sum(r['success'] for r in good),'paired':len(paired),
                     'rescues':rescues,'spoils':spoils,'net':rescues-spoils,
                     'success_rate':success_rate,'success_rate_se':success_se,'distinct_starting_states':n_states,
                     'paired_success_change':paired_change,'paired_success_change_se':paired_se,
                     'paired_distinct_starting_states':n_paired_states,
                     'rescue_rate_given_zero_failure':rescue_rate,'rescue_rate_se':rescue_se,
                     'spoil_rate_given_zero_success':spoil_rate,'spoil_rate_se':spoil_se,
                     'total_pcp_interventions':sum(r.get('n_corrections',0) for r in good)})
    return sorted(rows,key=lambda r:(r['completed']>0,r['errors']==0,r['net'],r['success_rate'] or 0.,r['arm_id']),reverse=True)


def cuda_oom_message(error):
    message=str(error)
    lower=message.lower()
    if isinstance(error,torch.cuda.OutOfMemoryError) or ('out of memory' in lower and 'cuda' in lower):
        return message
    return None


def guarded_cuda_batch(operation):
    """Release the exception traceback before attempting allocator/context cleanup."""
    try:
        results=operation()
    except Exception as error:
        message=cuda_oom_message(error)
        if message is None:raise
        return None,message
    for result in results:
        message=cuda_oom_message(result.get('error_msg',''))
        if message is not None:return None,message
    return results,None


def run_sweep(*,checkpoint_paths,worker_index,worker_count=2,batch_size=64,
              simulator_processes=8,output_dir='/content/pcp-search',experiment=EXPERIMENT,
              grid_override=None,tap_type=SweepTap,protocol=None):
    """Finite large search plan; no wall-clock budget or automatic paid dispatch."""
    from . import models
    from .rollout import run_episode_batch
    from .store import SupabaseStore,gather_provenance
    if not 0<=worker_index<worker_count:raise ValueError('worker index')
    if batch_size<2:raise ValueError('batch_size must be >=2')
    torch.set_num_threads(2)
    from .smolvla_jeff_replication import configure_precision
    configure_precision()
    device='cuda'
    critics={};shas={}
    for name,path in checkpoint_paths.items():critics[name],shas[name]=load_critic(path,device)
    if set(critics)!= {'frozen_mc','cnn_mc','td'}:raise ValueError('Supply all three critic checkpoints')
    cases=prepare_cases();targets=sorted({min(n,len(cases)) for n in RUNGS});grid=sweep_grid() if grid_override is None else grid_override;arms=[a for a in grid if a['ordinal']%worker_count==worker_index]
    cfg=RolloutConfig(pnp_steps=(1,2,3),pnp_k=3,pnp_k_by_step=(3,1,1),refine=True,
                      n_action_steps=10,num_inference_steps=10,skip_unused_renders=True,
                      render_lead=2,save_trajectory=True,video='off')
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest={'experiment':experiment,'checkpoint_shas':shas,'arms':grid,'rungs':targets,
              'seed_streams':[0,1,2],'precision':{'tf32':False,'matmul':'highest'},
              'batch_grouping':'same task and historical two-shard identity parity; no mixed-task text padding',
              'allocation':'equal exposure for every predefined configuration','worker_count':worker_count,'code_sha256':code_hash,
              'cases':[dict(suite=e['suite'],task_idx=e['task_idx'],ep_idx=e['ep_idx'],
                            init_state_hash=e['init_state_hash'],behavior_seed_index=e['behavior_seed_index']) for e in cases],
              'units':'common frozen-MC-training standardized clean action RMS; total trust radius across inner iterations',
              'cnn_anchor':'current probe clean estimate, detached and held fixed throughout inner search',
              'interpretation':'fixed equal-exposure exploratory full-episode evaluation; benchmark states may have been used previously',
              'baseline':'stock SmolVLA: no P&P probes, no re-noising, no Q correction; 10 Euler steps, execute 10 actions',
              'treatment_parent':'P&P steps1/2/3 K3/1/1; Q/search interventions added to that parent',
              'no_time_cutoff':True}
    if protocol is not None:
        manifest['proposal_protocol']=protocol
        manifest['proposal_code_sha256']=hashlib.sha256(Path(__import__(tap_type.__module__,fromlist=['']).__file__).read_bytes()).hexdigest()
    manifest=json.loads(json.dumps(manifest))
    manifest['manifest_hash']=digest(manifest)
    store=SupabaseStore();prefix=f'smolvla_pcp_sweep/{experiment}/{manifest["manifest_hash"]}'
    manifest_key=f'{prefix}/manifest.json'
    try:
        existing=json.loads(store.client.storage.from_(store.bucket).download(manifest_key))
    except Exception as error:
        if not any(x in str(error).lower() for x in ('404','not found','does not exist')):raise
        store._upload(manifest_key,json.dumps(manifest).encode());existing=manifest
    if existing!=manifest:raise ValueError('immutable manifest differs')
    policy,preprocess,postprocess=models.load_smolvla(device=device)
    store.start_run('smolvla_pcp_a100_search','libero',experiment,config={
        **manifest,'worker_index':worker_index,'batch_size':batch_size,'simulator_processes':simulator_processes},
        provenance=gather_provenance(model_repo_id='HuggingFaceVLA/smolvla_libero'))
    # Independent per-worker rows and artifacts; each owns its zero baseline.
    rows=store.fetch_all('rollouts','rollout_id,status,success,ms_candidate_u',
                        configure=lambda q:q.eq('experiment',experiment).eq('ms_candidate_u->q_guidance->>worker_index',str(worker_index)).eq('ms_candidate_u->q_guidance->>manifest_hash',manifest['manifest_hash']),order_by=('rollout_id',))
    records=[]
    for row in rows:
        info=(row.get('ms_candidate_u') or {}).get('q_guidance') or {}
        if info.get('worker_index')==worker_index and info.get('manifest_hash')==manifest['manifest_hash']:
            records.append(dict(arm_id=info['arm_id'],case_index=info['case_index'],success=row['success'],status=row['status'],rollout_id=row['rollout_id'],n_corrections=info.get('n_corrections',0)))
    done={(r['arm_id'],r['case_index']) for r in records if r['status']=='completed'}
    records={(r['arm_id'],r['case_index']):r for r in records}
    planned_rollouts=(len(arms)+1)*len(cases)
    actors=[EnvActor() for _ in range(min(simulator_processes,batch_size))]
    step_pool=ThreadPoolExecutor(max_workers=batch_size)
    uploads=ThreadPoolExecutor(max_workers=4);pending=[];thread_state=threading.local()
    gpu_samples=[];sample_stop=threading.Event()
    def sample_gpu():
        while not sample_stop.wait(10):
            try:
                data=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                gpu_samples.append([float(x.strip()) for x in data.splitlines()[0].split(',')])
            except Exception:pass
    sampler=threading.Thread(target=sample_gpu,daemon=True);sampler.start()
    started=time.monotonic();completed_new=0;status='failed'
    def uploader(ep,method,rid,result,telemetry):
        if not hasattr(thread_state,'store'):
            thread_state.store=store.fork_for_thread();thread_state.store.experiment=experiment;thread_state.store.run_id=store.run_id
        remote=thread_state.store
        remote._upload(f'{prefix}/worker_{worker_index}/chunks/{rid}.json',json.dumps(telemetry,allow_nan=False).encode())
        remote.log_result(rid,ep,method,cfg,result,persist_probe_rows=False)
    def flush(all_pending=False):
        while pending and (all_pending or len(pending)>=2*batch_size or pending[0].done()):pending.pop(0).result()
    def publish(stage):
        flush(True)
        board=leaderboard(arms,records,stage,cases)
        report={'worker_index':worker_index,'manifest_hash':manifest['manifest_hash'],'stage_cases':stage,
                'completed_new':completed_new,'completed_total':len(done),'elapsed_hours':(time.monotonic()-started)/3600,
                'rollouts_per_hour':completed_new/max((time.monotonic()-started)/3600,1e-9),
                'batch_size':batch_size,'planned_rollouts_upper':planned_rollouts,
                'remaining_hours_at_current_rate':max(0,planned_rollouts-len(done))/max(completed_new/max((time.monotonic()-started)/3600,1e-9),1e-9) if completed_new else None,
                'gpu_utilization_mean':float(np.mean(gpu_samples,axis=0)[0]) if gpu_samples else None,
                'gpu_memory_used_mib_mean':float(np.mean(gpu_samples,axis=0)[1]) if gpu_samples else None,
                'peak_torch_memory_gib':torch.cuda.max_memory_allocated()/2**30,'leaderboard':board}
        text=json.dumps(report,indent=2)
        (directory/f'worker_{worker_index}_latest.json').write_text(text)
        store._upload(f'{prefix}/worker_{worker_index}/latest.json',text.encode())
        store._upload(f'smolvla_pcp_sweep/{experiment}/worker_{worker_index}_latest.json',text.encode())
        print(json.dumps({k:v for k,v in report.items() if k!='leaderboard'}),flush=True)
        print('Leaders:',[(r['arm_id'],r['successes'],r['completed'],r['net']) for r in board[:5]],flush=True)
        return board
    def collect(arm,target):
        nonlocal completed_new,batch_size
        todo=[i for i in range(target) if (arm['arm_id'],i) not in done]
        group_key=lambda i:(cases[i]['suite'],cases[i]['task_idx'],cases[i]['ep_idx']%2,cases[i]['behavior_seed_index'])
        todo.sort(key=lambda i:(group_key(i),cases[i]['ep_idx']))
        offset=0
        while offset<len(todo):
            ids=[]
            for i in todo[offset:offset+batch_size]:
                if group_key(i)!=group_key(todo[offset]):break
                ids.append(i)
            eps=[cases[i] for i in ids]
            telemetry=[[] for _ in ids]
            factory=lambda **kw:tap_type(arm=arm,critic=critics.get(arm['critic']),
                correction_std=critics['frozen_mc'].action_std,
                emit=lambda lane,item:telemetry[lane].append(item),**kw)
            arm_config,factory=rollout_arm_setup(arm,cfg,factory)
            envs=[EnvProxy(actors[lane%len(actors)],lane,e['bddl_path']) for lane,e in enumerate(eps)]
            results,oom=guarded_cuda_batch(lambda:run_episode_batch(
                envs,eps,policy,preprocess,postprocess,device,arm_config,
                tap_factory=factory,env_step_executor=step_pool))
            if oom is not None:
                if batch_size<=2:raise RuntimeError('CUDA OOM at batch size 2: '+oom)
                previous=batch_size;batch_size=max(2,batch_size//2)
                # MuJoCo EGL contexts allocate GPU memory outside PyTorch. Simply
                # emptying the Torch cache leaves the oversized simulator pool resident.
                from . import sampler
                sampler.set_strategy(policy.model,None)
                telemetry.clear();envs.clear()
                for actor in actors:actor.close()
                actors.clear();gc.collect();torch.cuda.empty_cache()
                actors.extend(EnvActor() for _ in range(min(simulator_processes,batch_size)))
                print(f'[CUDA OOM] reduced batch {previous} -> {batch_size}; rebuilt simulator contexts; retrying same cases',flush=True)
                continue
            for case,ep,result,chunks in zip(ids,eps,results,telemetry):
                method=f'pcp_search_w{worker_index}_{arm["arm_id"]}_{manifest["manifest_hash"][:12]}'
                rid=store.rollout_id(experiment,ep,method,cfg)
                info={'manifest_hash':manifest['manifest_hash'],'worker_index':worker_index,
                      'case_index':case,'arm_id':arm['arm_id'],'arm':arm,
                      'batch_performance':result.get('batch_performance'),
                      'chunk_telemetry_path':f'{prefix}/worker_{worker_index}/chunks/{rid}.json',
                      'n_corrections':sum(x.get('intervened',x['standardized_rms']>0) for x in chunks),
                      'bad_gradient_chunks':sum(x['bad_gradient'] for x in chunks),
                      'n_searches':sum(x.get('search_open',False) for x in chunks),
                      'n_reranked_candidates':sum(x.get('n_candidates',0) for x in chunks)}
                result['recorder_episode']=None;result['q_guidance_telemetry']=info
                pending.append(uploads.submit(uploader,ep,method,rid,result,chunks))
                record=dict(arm_id=arm['arm_id'],case_index=case,success=bool(result['success']),status=result['status'],rollout_id=rid,n_corrections=info['n_corrections'])
                records[(arm['arm_id'],case)]=record
                with (directory/f'worker_{worker_index}_journal.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if result['status']=='completed':done.add((arm['arm_id'],case));completed_new+=1
                else:print('Episode error:',arm['arm_id'],case,result['error_msg'],flush=True)
            offset+=len(ids);flush()
            if offset%(4*batch_size)==0 or offset>=len(todo):publish(target)
            if any(result['status']!='completed' for result in results):
                flush(True)
                first=next(result for result in results if result['status']!='completed')
                raise RuntimeError('Episode batch failed; errors saved to Supabase. Fix before resuming: '+str(first.get('error_msg')))
    try:
        for target in targets:
            collect(baseline_arm(),target)
            if sum(arm=='zero' and case<target for arm,case in done)!=target:
                raise RuntimeError('Incomplete zero baseline; inspect logged errors and resume')
            for number,arm in enumerate(arms):
                collect(arm,target)
                if (number+1)%4==0:flush();publish(target)
            flush(True);board=publish(target)
        status='completed'
    finally:
        sample_stop.set();sampler.join(6)
        try:flush(True)
        finally:
            uploads.shutdown(wait=True);step_pool.shutdown(wait=True)
            for actor in actors:actor.close()
            store.finish_run(status=status,n_rollouts=completed_new)
    return publish(targets[-1])


def pull_progress(experiment=EXPERIMENT, worker_count=2):
    """Small online summaries; all individual rows remain queryable in rollouts."""
    from .store import SupabaseStore
    store=SupabaseStore();reports=[]
    for worker in range(worker_count):
        key=f'smolvla_pcp_sweep/{experiment}/worker_{worker}_latest.json'
        try:reports.append(json.loads(store.client.storage.from_(store.bucket).download(key)))
        except Exception as error:
            if not any(t in str(error).lower() for t in ('404','not found','does not exist')):raise
    return reports
