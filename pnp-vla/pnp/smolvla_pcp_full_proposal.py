"""Frozen-policy PCP: disagreement gate, bounded correction, regeneration and terminal Q ranking."""
from __future__ import annotations
import itertools
import math
import torch
from .smolvla_pcp_sweep import SweepTap, bounded_move, digest, run_sweep
from .smolvla_anchored_critic import AnchoredCritic
from .qplanning_critic.model import pool_prefix_tokens
from .pnp import run_probe

EXPERIMENT='smolvla-pcp-full-proposal-jeff-matched-v4'
PROTOCOL={
 'search_steps':[2,5,8],'candidates':3,'uncertainty_probes':3,
 'uncertainty':'RMS disagreement of one-step clean estimates in common frozen-MC action-standardized coordinates; not calibrated epistemic uncertainty',
 'threshold':.06,'threshold_selection':'fixed heuristic, not fitted to candidate outcomes',
 'ranking':'Q of fully denoised candidates; include exact zero-correction continuation as candidate zero',
 'noise':'fresh independent branch noise, deterministic by episode/chunk/branch, shared across full-proposal arms',
 'correction':'first ten executable actions only; same radius convention as workers 0/1',
 'scope':'frozen MC and continued TD; CNN excluded from this focused experiment',
 'batching':'each branch is GPU-batched over episodes; shared prefix KV cache, branches evaluated sequentially',
 'baseline':'stock SmolVLA: no P&P and no Q correction',
 'search_parent':'P&P steps1/2/3 K3/1/1; candidate-zero fallback is the uncorrected P&P continuation, distinct from the stock evaluation control',
}


def proposal_grid():
    arms=[]
    def add(critic,step,radius,threshold,variant):
        arms.append(dict(critic=critic,step=step,radius=radius,every=1,gate=1.01,
                         iterations=1,mode='ascent',uncertainty_threshold=threshold,
                         candidates=3,variant=variant))
    for critic,step,threshold in itertools.product(('frozen_mc','td'),(2,5,8),(0.,.06)):
        add(critic,step,.06,threshold,'full')
    for critic,step in itertools.product(('frozen_mc','td'),(2,5,8)):
        add(critic,step,0.,0.,'rerank_only')
    # Controls first at each equal-exposure stage, then interleave critics.
    arms.sort(key=lambda a:(a['variant']=='full',digest(a)))
    for ordinal,arm in enumerate(arms):arm.update(ordinal=ordinal,arm_id=digest(arm)[:16])
    return arms


def denoise_remaining(x,s,step,num_steps,vfield,probe_noises=None):
    dt=-1./num_steps
    for index in range(step,num_steps):
        level=1.+index*dt
        for noise in (probe_noises or {}).get(index,[]):
            estimate=x-level*vfield(x,level)
            x=(1-level)*estimate+level*noise
        x=x+dt*vfield(x,level)
    return x


def choose_candidates(candidates,scores,search_open):
    """Candidates [N+1,B,...]; candidate zero is the unmodified continuation."""
    scores=torch.where(torch.isfinite(scores),scores,torch.full_like(scores,-torch.inf))
    selected=scores.argmax(0)
    # Ties favor candidate zero; closed gates preserve baseline exactly.
    selected=torch.where(search_open,selected,torch.zeros_like(selected))
    return candidates[selected,torch.arange(candidates.shape[1],device=candidates.device)],selected


class FullProposalTap(SweepTap):
    def selected(self,step,s):
        return super().selected(step,s) or (self.arm['mode']!='zero' and step==self.arm['step'])

    def complete_chunk(self,x_t,s,vf,timed_vf,ctx):
        if self.arm['mode']=='zero' or ctx.step!=self.arm['step']:return None
        baseline_selected=self.config.probe_selected(ctx.step,float(s))
        # Added search times must not alter the parent RNG or closed-gate baseline.
        probe_generators=self.generators
        if not baseline_selected:
            probe_generators=[]
            for generator in self.generators:
                clone=torch.Generator(device=x_t.device);clone.set_state(generator.get_state())
                probe_generators.append(clone)
        probe=run_probe(x_t,s,vf,k=self.config.probe_k(ctx.step) if baseline_selected else 1,
                        adim=self.adim,generators=probe_generators,record_telemetry=False)
        baseline_start=probe.x_acc if baseline_selected else x_t
        # Replay any remaining P&P steps with identical draws in every candidate.
        remaining_noises={}
        for index in range(ctx.step+1,ctx.num_steps):
            level=1.-index/ctx.num_steps
            if self.config.probe_selected(index,level):
                remaining_noises[index]=[]
                for _ in range(self.config.probe_k(index)):
                    noise=torch.empty_like(x_t)
                    for lane,generator in enumerate(self.generators):noise[lane].normal_(generator=generator)
                    remaining_noises[index].append(noise)
        full=probe.z_hat_full.detach();anchor=full[:,:10,:7].clone()
        valid=torch.arange(10,device=full.device)[None]<torch.tensor(self.remaining,device=full.device)[:,None]
        std=self.correction_std
        prefix,pad=pool_prefix_tokens(ctx.prefix_embeddings.detach(),ctx.prefix_pad_masks.detach(),128)
        if isinstance(self.critic,AnchoredCritic):
            features,base=self.critic.reference_features(prefix,pad,self.robot,self.proprio,anchor,valid)
            score=lambda a:self.critic.logits_from_features(features,base,a,valid,anchor).reshape(-1).sigmoid()
        else:
            score=lambda a:self.critic(prefix,pad,self.robot,self.proprio,a,valid).reshape(-1).sigmoid()
        noises=[];probe_actions=[]
        for branch in range(3):
            noise=torch.empty_like(full)
            for lane,chunk in enumerate(self.chunk_indices):
                seed=(int(self.generators[lane].initial_seed()) ^ 0x5F3759DF ^ ((int(chunk)+1)*1000003) ^ ((branch+1)*15485863))%(2**63-1)
                noise[lane].normal_(generator=torch.Generator(device=full.device).manual_seed(seed))
            noises.append(noise)
            x=(1-float(s))*full+float(s)*noise
            probe_actions.append((x-float(s)*vf(x))[:,:10,:7])
        samples=torch.stack(probe_actions)/std
        denominator=(valid.sum(1)*7).clamp_min(1)
        spread=((samples-samples.mean(0)).square().mean(0)*valid[...,None]).sum((1,2))
        uncertainty=(spread/denominator).sqrt()
        search_open=uncertainty>=self.arm['uncertainty_threshold']
        candidate=anchor.clone();bad=torch.zeros_like(search_open)
        with torch.no_grad():q_anchor=score(anchor)
        search_open &= torch.isfinite(uncertainty)&torch.isfinite(q_anchor)
        if self.arm['radius']>0:
            with torch.enable_grad():
                candidate=anchor.clone().requires_grad_(True)
                q=score(candidate)
                gradient,=torch.autograd.grad(torch.nan_to_num(q).sum(),candidate)
            candidate,usable=bounded_move(candidate.detach(),anchor,gradient.detach(),valid,std,self.arm['radius'])
            bad=search_open&~usable
        candidate=torch.where(search_open[:,None,None],candidate,anchor)
        delta=candidate.detach()-anchor
        corrected=full.clone();corrected[:,:10,:7]+=delta
        # Complete the exact uncorrected path first; it is also the closed-gate result.
        baseline=denoise_remaining(baseline_start,float(s),ctx.step,ctx.num_steps,timed_vf,remaining_noises)
        candidates=[baseline]
        with torch.no_grad():scores=[score(baseline[:,:10,:7])]
        # All active lanes remain batched; no policy weights or prefix are recomputed.
        if bool(search_open.any()):
            for branch in range(self.arm['candidates']):
                start=(1-float(s))*corrected+float(s)*noises[branch]
                terminal=denoise_remaining(start,float(s),ctx.step,ctx.num_steps,timed_vf,remaining_noises)
                candidates.append(terminal)
                with torch.no_grad():scores.append(score(terminal[:,:10,:7]))
        stacked=torch.stack(candidates);q_scores=torch.stack(scores)
        chosen,indices=choose_candidates(stacked,q_scores,search_open)
        native=(delta.square().sum((1,2))/denominator).sqrt()
        standardized=((delta/std).square().sum((1,2))/denominator).sqrt()
        values=torch.stack([uncertainty,native,standardized,search_open.float(),bad.float(),indices.float()],1).detach().cpu().tolist()
        q_values=q_scores.detach().cpu().T.tolist()
        qa=q_anchor.detach().cpu().tolist()
        for lane,(row,qs) in enumerate(zip(values,q_values)):
            clean=lambda z:float(z) if math.isfinite(float(z)) else None
            u,native_rms,rms,opened,invalid,index=row
            index=int(index)
            self.emit(self.lane_ids[lane],dict(chunk_index=self.chunk_indices[lane],euler_step=ctx.step,
                noise_level=float(s),flow_progress=1-float(s),
                uncertainty=clean(u),uncertainty_threshold=self.arm['uncertainty_threshold'],
                native_rms=native_rms,standardized_rms=rms,gate_open=bool(opened),search_open=bool(opened),
                bad_gradient=bool(invalid),q_before=clean(qa[lane]),q_after=clean(qs[index]),
                candidate_q=[clean(q) for q in qs],selected_candidate=index,
                n_candidates=len(qs)-1 if opened else 0,intervened=bool(opened and index>0)))
        return chosen


def run_full_proposal(**kwargs):
    return run_sweep(**kwargs,experiment=EXPERIMENT,grid_override=proposal_grid(),
                     tap_type=FullProposalTap,protocol=PROTOCOL)
