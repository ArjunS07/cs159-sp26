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
