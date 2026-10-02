from types import SimpleNamespace
import numpy as np
import torch
from pnp.config import RolloutConfig
from pnp.pnp import PnPRecorder,run_probe
from pnp.smolvla_pcp_full_proposal import proposal_grid,FullProposalTap,choose_candidates,denoise_remaining


def test_grid_is_small_and_has_isolating_controls():
    arms=proposal_grid()
    assert len(arms)==18 and len({a['arm_id'] for a in arms})==18
    assert sum(a['variant']=='full' for a in arms)==12
    assert sum(a['variant']=='rerank_only' for a in arms)==6
    assert {a['step'] for a in arms}=={2,5,8}


def test_reranking_has_baseline_fallback_closed_gates_and_nan_handling():
    actions=torch.arange(6.).reshape(3,2,1)
    scores=torch.tensor([[.4,.8],[.7,.9],[float('nan'),.6]])
    chosen,idx=choose_candidates(actions,scores,torch.tensor([True,False]))
    assert idx.tolist()==[1,0] and chosen[:,0].tolist()==[2.,1.]
    _,idx=choose_candidates(actions,torch.ones(3,2),torch.ones(2,dtype=torch.bool))
    assert idx.tolist()==[0,0]


class Critic(torch.nn.Module):
    def forward(self,prefix,pad,robot,proprio,action,valid):
        return (action*valid[...,None]).sum((1,2))[:,None]*.1


def make_tap(threshold,radius=.06):
    obs={'agentview_image':np.zeros((4,4,3),np.uint8),'robot0_eye_in_hand_image':np.zeros((4,4,3),np.uint8),
         'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1.]),'robot0_gripper_qpos':np.zeros(2)}
    recorders=[PnPRecorder(),PnPRecorder()]
    for recorder in recorders:recorder.new_episode()
    gens=[torch.Generator().manual_seed(i+10) for i in range(2)]
    arm={**proposal_grid()[0],'step':3,'radius':radius,'uncertainty_threshold':threshold,'candidates':3}
    telemetry=[]
    tap=FullProposalTap(arm=arm,critic=Critic(),correction_std=torch.ones(7),
        emit=lambda lane,row:telemetry.append(row),episodes=[{'max_steps':20,'task_desc':'test'}]*2,
        observations=[obs,obs],steps=[0,18],chunk_indices=[0,0],lane_ids=[0,1],
        config=RolloutConfig(pnp_steps=(1,2,3),pnp_k=3,pnp_k_by_step=(3,1,1),refine=True),
        recorders=recorders,seeds=gens,device='cpu',adim=7)
    return tap,telemetry


def test_closed_gate_exactly_preserves_baseline_terminal_and_rng():
    tap,telemetry=make_tap(1e6)
    x=torch.zeros(2,50,7);vf=lambda x:x*.2
    ctx=SimpleNamespace(step=3,num_steps=10,prefix_embeddings=torch.zeros(2,4,3),prefix_pad_masks=torch.ones(2,4,dtype=torch.bool))
    gens=[torch.Generator().manual_seed(i+10) for i in range(2)]
    probe=run_probe(x,.7,vf,k=1,generators=gens,record_telemetry=False)
    expected=denoise_remaining(probe.x_acc,.7,3,10,lambda x,s:vf(x))
    actual=tap.complete_chunk(x,.7,vf,lambda x,s:vf(x),ctx)
    assert torch.equal(actual,expected)
    assert all(torch.equal(a.get_state(),b.get_state()) for a,b in zip(gens,tap.generators))
    assert all(not row['search_open'] and row['selected_candidate']==0 for row in telemetry)


def test_open_gate_ranks_final_actions_and_bounds_injected_correction():
    tap,telemetry=make_tap(0.)
    ctx=SimpleNamespace(step=3,num_steps=10,prefix_embeddings=torch.zeros(2,4,3),prefix_pad_masks=torch.ones(2,4,dtype=torch.bool))
    out=tap.complete_chunk(torch.zeros(2,50,7),.7,lambda x:x*.2,lambda x,s:x*.2,ctx)
    assert torch.isfinite(out).all()
    for row in telemetry:
        assert row['n_candidates']==3 and row['standardized_rms']<=.060001
        assert row['q_after']==max(row['candidate_q'])
        assert row['q_after']>=row['candidate_q'][0]


def test_sampler_calls_full_proposal_and_returns_its_terminal_winner(monkeypatch):
    import sys,types
    from pnp.sampler import install_smolvla_patch
    fake=types.ModuleType('lerobot.policies.smolvla.modeling_smolvla')
    fake.make_att_2d_masks=lambda pad,att:pad[:,:,None]&pad[:,None,:]
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    tap,telemetry=make_tap(0.)
    model=SimpleNamespace(config=SimpleNamespace(num_steps=10,chunk_size=50,max_action_dim=7,use_cache=True),
        sample_actions=lambda *a,**k:torch.zeros(2,50,7),
        embed_prefix=lambda *a,**k:(torch.zeros(2,4,3),torch.ones(2,4,dtype=torch.bool),torch.ones(2,4,dtype=torch.bool)),
        vlm_with_expert=SimpleNamespace(forward=lambda **k:(None,None)),
        denoise_step=lambda **k:k['x_t']*.2)
    install_smolvla_patch(model);model._pnp.strategy=tap
    result=model.sample_actions([],[],torch.zeros(2,4,dtype=torch.long),torch.ones(2,4,dtype=torch.bool),
                                torch.zeros(2,8),noise=torch.zeros(2,50,7))
    assert result.shape==(2,50,7) and torch.isfinite(result).all()
    assert len(telemetry)==2 and all(row['n_candidates']==3 for row in telemetry)


def test_simulator_replaces_old_task_before_allocating_new_one(monkeypatch):
    from pnp import libero_env
    from pnp.smolvla_pcp_sweep import _env_server
    events=[]
    class Env:
        def __init__(self,path):self.path=path;events.append(('make',path))
        def reset(self):return {'robot0_eef_pos':np.zeros(3)}
        def check_success(self):return False
        def close(self):events.append(('close',self.path))
    class Pipe:
        def __init__(self):self.requests=iter([(0,'a','reset',(),True),(0,'b','reset',(),True),None])
        def recv(self):return next(self.requests)
        def send(self,value):assert value[0]
        def close(self):pass
    monkeypatch.setattr(libero_env,'make_env',Env)
    monkeypatch.setattr(libero_env,'set_camera_observables',lambda *args:True)
    _env_server(Pipe())
    assert events==[('make','a'),('close','a'),('make','b'),('close','b')]


def test_closed_gate_preserves_parent_at_early_middle_and_late_search_times():
    for step in (2,5,8):
        tap,telemetry=make_tap(1e6);tap.arm['step']=step
        x=torch.ones(2,50,7)*.1
        ctx=SimpleNamespace(step=step,num_steps=10,prefix_embeddings=torch.zeros(2,4,3),prefix_pad_masks=torch.ones(2,4,dtype=torch.bool))
        generators=[torch.Generator().manual_seed(i+10) for i in range(2)]
        expected=x.clone()
        for index in range(step,10):
            level=1.-index/10
            if index in (1,2,3):
                expected=run_probe(expected,level,lambda x:x*.2,k=1,generators=generators,record_telemetry=False).x_acc
            expected=expected-.1*expected*.2
        actual=tap.complete_chunk(x,1-step/10,lambda x:x*.2,lambda x,s:x*.2,ctx)
        assert torch.equal(actual,expected)
        assert all(torch.equal(a.get_state(),b.get_state()) for a,b in zip(generators,tap.generators))
        assert tap.selected(step,1-step/10)
