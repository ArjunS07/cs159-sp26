from pnp.smolvla_jeff_replication import compare_rows

def rows():
    return [dict(suite='s',task_idx=i//10,episode_idx=i%10,init_state_hash=str(i),
                 status='completed',success=i<255,episode_seed=i,n_steps=100) for i in range(400)]

def test_equal_totals_are_not_episode_replication():
    old=rows();new=[dict(r) for r in old]
    new[0]['success']=False;new[399]['success']=True
    report=compare_rows(old,new)
    assert report['replicated_successes']==report['historical_successes']==255
    assert report['outcome_matches']==398 and not report['outcomes_replicated']

def test_complete_exact_outcomes_and_seeds_required():
    old=rows()
    assert compare_rows(old,old)['outcomes_replicated']
    assert not compare_rows(old,old[:-1])['outcomes_replicated']
    new=[dict(r) for r in old];new[0]['episode_seed']+=1
    assert not compare_rows(old,new)['outcomes_replicated']


def test_historical_plan_retains_resume_companions_and_rendering():
    from pnp.smolvla_jeff_replication import historical_batch_plan, identity
    eps = [dict(suite='s',task_idx=0,ep_idx=i,init_state_hash=str(i)) for i in range(6)]
    def row(i,run): return dict(eps[i],episode_idx=i,run_id=run)
    # Source A saved only episode 0 from batch (0,2). Its companion 2 is later saved by B.
    ref=[row(0,'A'),row(2,'B'),row(4,'B'),row(1,'C'),row(3,'C'),row(5,'C')]
    def run(r,t,shard,size,sha):
        return dict(run_id=r,created_at=t,pnp_git_sha=sha,
                    config_json=dict(shard_count=2,shard_index=shard,rollout_batch_size=size))
    runs=[run('C','3',1,8,'8ad4afe'),run('B','2',0,8,'8ad4afe'),run('A','1',0,2,'e9d92a5')]
    plan=historical_batch_plan(eps,ref,runs)
    assert [e['ep_idx'] for e in plan[0]['members']]==[0,2]
    assert plan[0]['targets']==[identity(eps[0])]
    assert not plan[0]['skip_unused_renders']
    assert [e['ep_idx'] for e in plan[1]['members']]==[2,4]
    assert plan[1]['skip_unused_renders']
    assert len([x for b in plan for x in b['targets']])==6


def test_historical_plan_rejects_incomplete_provenance():
    import pytest
    from pnp.smolvla_jeff_replication import historical_batch_plan
    eps=[dict(suite='s',task_idx=0,ep_idx=0,init_state_hash='a')]
    ref=[dict(eps[0],run_id='missing')]
    with pytest.raises(RuntimeError,match='cover'):
        historical_batch_plan(eps,ref,[])


def test_serial_historical_batch_retains_sparse_rendering():
    from pnp.smolvla_jeff_replication import historical_batch_plan
    eps=[dict(suite='s',task_idx=0,ep_idx=0,init_state_hash='a')]
    ref=[dict(eps[0],run_id='A')]
    runs=[dict(run_id='A',created_at='1',pnp_git_sha='e9d92a5',
               config_json=dict(shard_count=2,shard_index=0,rollout_batch_size=2))]
    # The original serial path already honored sparse rendering; only its batched path did not.
    assert historical_batch_plan(eps,ref,runs)[0]['skip_unused_renders']
