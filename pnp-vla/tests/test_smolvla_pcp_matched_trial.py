from pnp.smolvla_pcp_matched_trial import arm_config, paired_summary


def test_stock_has_no_probe_and_pcp_has_fixed_parent():
    stock=arm_config('stock',{'skip_unused_renders':False})
    pcp=arm_config('pcp',{'skip_unused_renders':False})
    assert not stock.has_probe and not stock.refine
    assert pcp.has_probe and pcp.refine
    assert pcp.pnp_steps==(1,2,3) and pcp.pnp_k_by_step==(3,1,1)
    assert stock.n_action_steps==pcp.n_action_steps==10
    assert not stock.skip_unused_renders and not pcp.skip_unused_renders


def test_paired_summary_uses_only_completed_shared_identities():
    def row(i,s,status='completed'):
        return dict(suite='s',task_idx=i//2,episode_idx=i,init_state_hash=str(i),
                    success=s,status=status,episode_seed=i)
    a=[row(0,False),row(1,True),row(2,True),row(3,False)]
    b=[row(0,True),row(1,False),row(2,True),row(3,True,'error')]
    r=paired_summary(a,b)
    assert r['paired']==3 and r['rescues']==r['spoils']==1
    assert r['stock_successes']==r['pcp_successes']==2
    assert r['rescue_rate']==1 and r['spoil_rate']==.5
    assert r['task_groups']==2 and r['seed_mismatches']==0
    assert r['paired_difference']==0 and not r['complete']
