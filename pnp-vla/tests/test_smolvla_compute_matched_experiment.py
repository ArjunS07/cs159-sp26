from pnp.config import Method
from pnp.smolvla_compute_matched_experiment import (
    SAMPLE_EQUIVALENT_NFES,
    build_smolvla_compute_matched_methods,
)


def test_compute_matched_methods_are_exactly_30_sample_equivalent_nfes():
    methods = build_smolvla_compute_matched_methods()
    assert [method for method, _ in methods] == [
        Method.SMOLVLA_STOCK_30_STEP,
        Method.SMOLVLA_THREE_STOCK_AVERAGE_10_STEP,
    ]
    stock, average = (config for _, config in methods)
    assert SAMPLE_EQUIVALENT_NFES == 30
    assert stock.num_inference_steps == 30
    assert stock.consensus_candidate_count is None
    assert average.consensus_candidate_count == 3
    assert average.consensus_candidate_inference_steps == 10
    assert average.consensus_projection_k is None
    assert average.consensus_projection_step is None
    assert all(config.n_action_steps == 10 for _, config in methods)
