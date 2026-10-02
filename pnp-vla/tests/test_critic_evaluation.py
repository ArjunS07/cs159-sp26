import numpy as np
import pytest

from pnp.critic_evaluation import bellman_summary, recorded_policy_residual
from scripts.audit_smolvla_critics import validate_alignment
from scripts.evaluate_smolvla_bellman import transition_errors, summarize_episodes
from scripts.summarize_smolvla_critic_uncertainty import auc_episode_bootstrap


def test_terminal_success_has_no_bootstrap_and_failure_is_zero():
    np.testing.assert_allclose(recorded_policy_residual([.8,.2,.4],[1,0,0],[0,0,1],[.9,.9,.6]),
                               [-.2,.2,-.2])
    with pytest.raises(ValueError,match="terminal masks"):
        recorded_policy_residual([.8],[1],[1],[.9])


def test_long_episodes_do_not_dominate_primary_error():
    result = bellman_summary([1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],["short","long","long","long"])
    assert result["episode_weighted"]["mae"] == .5
    assert result["transition_weighted"]["mae"] == .25
    assert result["episode_weighted"]["rmse"] == pytest.approx(np.sqrt(.5))


def test_same_roots_with_reordered_candidate_outcomes_are_rejected():
    with pytest.raises(ValueError,match="outcomes differ"):
        validate_alignment([{"group_id":"episode","labels":[False,True]}],{"episode":[True,False]})


def test_nstep_targets_stop_at_success_and_use_recorded_next_q():
    q=np.array([.2,.4,.7])
    one,terminal=transition_errors(q,[0,0,1],[1,1,0],1)
    five,all_terminal=transition_errors(q,[0,0,1],[1,1,0],5)
    np.testing.assert_allclose(one,[-.2,-.3,-.3])
    np.testing.assert_array_equal(terminal,[False,False,True])
    np.testing.assert_allclose(five,[-.8,-.6,-.3])
    assert all_terminal.all()


def test_episode_bellman_errors_weight_trajectories_equally_and_report_se():
    rows=[]
    for identity,errors in (("a",[0,1,1]),("b",[0,0,0])):
        for error in errors:
            rows.append(dict(group_id=identity,transitions=10,residual_mean=error,
                             residual_mae=error,residual_mse=error,mc_mse=error,
                             terminal_mse=error,nonterminal_mse=None))
    result=summarize_episodes(rows)
    assert result['episodes']==2 and result['trajectories']==6 and result['transitions']==60
    assert result['residual_mse']['mean']==pytest.approx(1/3)
    assert result['residual_mse']['standard_error']==pytest.approx(1/3)
    assert result['nonterminal_mse'] is None


def test_episode_bootstrap_auc_handles_tied_scores():
    y=np.array([[True,False],[False,True]])
    result=auc_episode_bootstrap(y,np.full((2,2),.5),replicates=50)
    assert result['mean']==.5 and result['standard_error']==0
    assert result['episodes']==2
