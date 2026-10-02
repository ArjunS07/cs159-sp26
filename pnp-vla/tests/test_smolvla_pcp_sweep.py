from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.config import RolloutConfig
from pnp.pnp import PnPRecorder,run_probe
from pnp.smolvla_pcp_sweep import bounded_move,sweep_grid,SweepTap,baseline_arm,leaderboard


def generators():return [torch.Generator().manual_seed(i+10) for i in range(2)]


def test_fast_probe_preserves_feedback_and_rng_exactly():
    x=torch.ones(2,50,7)
    for k in (1,3):
        left,right=generators(),generators()
        a=run_probe(x,.7,lambda t:t*.2,k=k,generators=left)
        b=run_probe(x,.7,lambda t:t*.2,k=k,generators=right,record_telemetry=False)
        for field in ('x_acc','z_hat_full','a_hats','last_eps'):
            assert torch.equal(getattr(a,field),getattr(b,field))
        assert all(torch.equal(g.get_state(),h.get_state()) for g,h in zip(left,right))


def test_projected_steps_obey_one_total_radius_and_mask():
    base=torch.zeros(2,10,7);valid=torch.ones(2,10,dtype=torch.bool);valid[1,3:]=False
    std=torch.linspace(.5,2,7);candidate=base
    for _ in range(3):candidate,_=bounded_move(candidate,base,torch.ones_like(base),valid,std,.06,1/3)
    rms=(((candidate-base)/std).square().sum((1,2))/(valid.sum(1)*7)).sqrt()
    assert torch.all(rms<=.060001)
    assert torch.all(candidate[1,3:]==0)


class ToyCritic(torch.nn.Module):
    def forward(self,prefix,pad,robot,proprio,action,valid):
        return (action*valid[...,None]).sum((1,2),keepdim=False)[:,None]*.02


def build_tap(arm):
    obs={'agentview_image':np.zeros((4,4,3),np.uint8),'robot0_eye_in_hand_image':np.zeros((4,4,3),np.uint8),
         'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1.]),'robot0_gripper_qpos':np.zeros(2)}
    eps=[{'max_steps':20,'task_desc':'test'} for _ in range(2)]
    recorders=[PnPRecorder(),PnPRecorder()]
    for r in recorders:r.new_episode()
    telemetry=[]
    tap=SweepTap(arm=arm,critic=ToyCritic(),correction_std=torch.ones(7),
                 emit=lambda lane,item:telemetry.append((lane,item)),
                 episodes=eps,observations=[obs,obs],steps=[0,18],chunk_indices=[0,0],lane_ids=[0,1],
                 config=RolloutConfig(pnp_steps=(1,2,3),pnp_k=3,pnp_k_by_step=(3,1,1),refine=True),
                 recorders=recorders,seeds=generators(),device='cpu',adim=7)
    return tap,telemetry


def test_actual_tap_ascent_changes_only_valid_prefix_and_raises_q():
    arm={**baseline_arm(),'mode':'ascent','radius':.06,'iterations':3}
    tap,telemetry=build_tap(arm)
    x=torch.zeros(2,50,7);vf=lambda t:t*.2
    context=SimpleNamespace(step=3,prefix_embeddings=torch.zeros(2,4,3),prefix_pad_masks=torch.ones(2,4,dtype=torch.bool))
    reference=run_probe(x,.7,vf,k=1,generators=generators(),record_telemetry=False)
    corrected=tap.step(x,.7,vf,context)
    assert torch.equal(corrected[:,10:],reference.x_acc[:,10:])
    assert torch.equal(corrected[1,2:10],reference.x_acc[1,2:10])
    assert all(row['q_after']>=row['q_before'] for _,row in telemetry)
    assert all(row['standardized_rms']<=.060001 for _,row in telemetry)


def test_zero_tap_is_exact_pnp_without_critic_call():
    tap,telemetry=build_tap(baseline_arm())
    x=torch.zeros(2,50,7);vf=lambda t:t*.2
    context=SimpleNamespace(step=3)
    reference=run_probe(x,.7,vf,k=1,generators=generators(),record_telemetry=False)
    assert torch.equal(tap.step(x,.7,vf,context),reference.x_acc)
    assert telemetry==[]


def test_grid_has_disjoint_workers_and_unique_arms():
    arms=sweep_grid();assert len(arms)==24
    shards=[{a['arm_id'] for a in arms if a['ordinal']%2==w} for w in (0,1)]
    assert len(shards[0])==len(shards[1])==12
    assert not shards[0]&shards[1]


def test_leaderboard_uses_paired_zero_and_retains_errors():
    arm=sweep_grid()[0]
    rows=[dict(arm_id='zero',case_index=0,status='completed',success=False),
          dict(arm_id=arm['arm_id'],case_index=0,status='completed',success=True),
          dict(arm_id=arm['arm_id'],case_index=1,status='errored',success=False)]
    board=leaderboard([arm],rows,2)[0]
    assert board['rescues']==1 and board['spoils']==0 and board['errors']==1


def test_clustered_se_does_not_treat_repeated_seeds_as_independent():
    from pnp.smolvla_pcp_sweep import clustered_ratio
    mean,se,n=clustered_ratio([1,1,0,0],[1,1,1,1],['a','a','b','b'])
    assert mean==.5 and se==.5 and n==2
    assert clustered_ratio([],[],[])==(None,None,0)


def test_cuda_backoff_guard_catches_raised_and_returned_oom_only():
    from pnp.smolvla_pcp_sweep import guarded_cuda_batch
    def raise_oom():raise RuntimeError('CUDA error: out of memory')
    assert guarded_cuda_batch(raise_oom)==(None,'CUDA error: out of memory')
    assert guarded_cuda_batch(lambda:[{'error_msg':'AcceleratorError: CUDA error: out of memory'}])[0] is None
    result=[{'status':'completed','error_msg':None}]
    assert guarded_cuda_batch(lambda:result)==(result,None)
    import pytest
    with pytest.raises(ValueError,match='unrelated'):
        guarded_cuda_batch(lambda:(_ for _ in ()).throw(ValueError('unrelated')))


def test_stock_control_bypasses_all_probes_and_custom_tap():
    from pnp.smolvla_pcp_sweep import rollout_arm_setup
    parent=RolloutConfig(pnp_steps=(1,2,3),pnp_k_by_step=(3,1,1),refine=True,
                         n_action_steps=10,num_inference_steps=10)
    factory=object()
    stock,tap=rollout_arm_setup(baseline_arm(),parent,factory)
    assert not stock.has_probe and not stock.refine and tap is None
    assert stock.n_action_steps==10 and stock.num_inference_steps==10
    assert parent.has_probe and parent.refine
    treatment,tap=rollout_arm_setup(sweep_grid()[0],parent,factory)
    assert treatment is parent and tap is factory
