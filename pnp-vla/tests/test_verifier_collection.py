from pnp.verifier.collection import (
    build_stratified_manifest, build_targeted_manifests, candidate_group_id,
    build_seeded_pro_manifest, collection_manifest_hash,
)


def test_candidate_group_id_is_stable_and_identity_sensitive():
    a = candidate_group_id("libero", "s", 1, 2, 3)
    assert a == "db118fe14552782aa0419a80"  # Legacy resume compatibility.
    assert a == candidate_group_id("libero", "s", 1, 2, 3)
    assert a != candidate_group_id("libero", "s", 1, 2, 4)
    assert a != candidate_group_id("libero", "s", 1, 2, 3, namespace="round-2")
    assert a != candidate_group_id(
        "libero", "s", 1, 2, 3, trajectory_seed=123)


def test_seeded_pro_manifest_is_deterministic_and_split_disjoint():
    rows = [{
        "rollout_id": f"r-{episode}", "benchmark": "libero_pro",
        "suite": f"suite-{episode % 2}", "task_idx": episode % 5,
        "episode_idx": episode, "chunk_idx": 2, "u_mean": episode,
        "success": episode % 3 != 0, "uncertainty_stratum": "high",
    } for episode in range(50)]
    first = build_seeded_pro_manifest(
        rows, development_target=24, test_target=16, seed=4)
    second = build_seeded_pro_manifest(
        list(reversed(rows)), development_target=24, test_target=16, seed=4)
    assert first == second
    development = {(row["suite"], row["task_idx"], row["episode_idx"],
                    row["trajectory_seed"]) for row in first["development"]}
    test = {(row["suite"], row["task_idx"], row["episode_idx"],
             row["trajectory_seed"]) for row in first["confirmatory_test"]}
    assert len(development) == 24 and len(test) == 16
    assert development.isdisjoint(test)
    assert {row["collection_split"] for row in first["development"]} == {
        "development"}
    test_suite_counts = {
        suite: sum(row["suite"] == suite for row in first["confirmatory_test"])
        for suite in {row["suite"] for row in first["confirmatory_test"]}}
    assert max(test_suite_counts.values()) - min(test_suite_counts.values()) <= 1


def test_manifest_uses_mixed_tasks_and_one_state_per_rollout():
    rollouts, steps = [], []
    for task in range(2):
        for episode in range(10):
            rid = f"r-{task}-{episode}"
            rollouts.append({"rollout_id": rid, "benchmark": "libero", "suite": "s",
                             "task_idx": task, "episode_idx": episode,
                             "success": episode < 5 if task == 0 else True})
            for chunk in range(2):
                steps.append({"rollout_id": rid, "chunk_idx": chunk,
                              "u_mean": episode + chunk / 10})
    manifest = build_stratified_manifest(rollouts, steps, targets={"libero": 6})
    assert len(manifest) == 6
    assert {row["task_idx"] for row in manifest} == {0}
    assert len({row["rollout_id"] for row in manifest}) == 6
    assert {row["uncertainty_stratum"] for row in manifest} == {"low", "mid", "high"}


def test_targeted_manifests_are_deterministic_disjoint_and_failure_enriched():
    rollouts, steps = [], []
    for benchmark in ("libero", "libero_pro"):
        for episode in range(30):
            rollout_id = f"{benchmark}-{episode}"
            rollouts.append({
                "rollout_id": rollout_id, "benchmark": benchmark, "suite": "s",
                "task_idx": episode % 3, "episode_idx": episode,
                "success": episode % 4 != 0,
            })
            for chunk in range(3):
                steps.append({
                    "rollout_id": rollout_id, "chunk_idx": chunk,
                    "u_mean": episode + chunk / 10,
                })
    excluded = {("libero", "s", 0, 0), ("libero_pro", "s", 0, 0)}
    kwargs = {
        "development_targets": {"libero": 6, "libero_pro": 6},
        "test_targets": {"libero": 3, "libero_pro": 3},
        "development_failure_fraction": .75,
        "seed": 9,
    }
    first = build_targeted_manifests(rollouts, steps, excluded, **kwargs)
    second = build_targeted_manifests(
        list(reversed(rollouts)), list(reversed(steps)), excluded, **kwargs)
    assert first == second
    assert collection_manifest_hash(first["development"]) == collection_manifest_hash(
        second["development"])
    development_ids = {
        (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"])
        for row in first["development"]}
    test_ids = {
        (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"])
        for row in first["test"]}
    assert len(development_ids) == 12
    assert len(test_ids) == 6
    assert development_ids.isdisjoint(test_ids | excluded)
    assert sum(not row["success"] for row in first["development"]) >= 4
    assert all(row["uncertainty_stratum"] == "high"
               for rows in first.values() for row in rows)


def test_targeted_manifest_uses_each_episodes_highest_uncertainty_state():
    rollouts = [{
        "rollout_id": f"r{episode}", "benchmark": "libero", "suite": "s",
        "task_idx": 0, "episode_idx": episode, "success": episode % 2 == 0,
    } for episode in range(20)]
    steps = [{
        "rollout_id": f"r{episode}", "chunk_idx": chunk,
        "u_mean": episode + chunk / 10,
    } for episode in range(20) for chunk in range(3)]
    manifests = build_targeted_manifests(
        rollouts, steps, set(),
        development_targets={"libero": 10},
        test_targets={"libero": 5},
        development_failure_fraction=.5,
    )
    selected = manifests["development"] + manifests["test"]
    assert len(selected) == 15
    assert len({row["episode_idx"] for row in selected}) == 15
    assert {row["chunk_idx"] for row in selected} == {2}


def test_targeted_manifest_can_accept_a_prospective_shortfall():
    rollouts = [{
        "rollout_id": f"r{i}", "benchmark": "libero", "suite": "s",
        "task_idx": 0, "episode_idx": i, "success": True,
    } for i in range(3)]
    steps = [{"rollout_id": f"r{i}", "chunk_idx": 0, "u_mean": i}
             for i in range(3)]
    manifests = build_targeted_manifests(
        rollouts, steps, set(), development_targets={"libero": 0},
        test_targets={"libero": 10}, allow_shortfall=True)
    assert len(manifests["test"]) == 3


def test_run_continuation_replans_every_n_action_steps():
    """The base-policy continuation is open-loop (full chunk) by default and closed-loop
    (execute n_action_steps, then replan) when n_action_steps is set."""
    import numpy as np
    import torch
    from unittest.mock import patch
    from pnp.verifier import collection as C

    class FakeEnv:
        def step(self, _action):
            return {}, 0.0, False, {}

        def check_success(self):
            return False

    class PNP:
        chunk_pos = -1.0

    class Model:
        _pnp = PNP()

    class Config:
        chunk_size = 3

    class Policy:
        model = Model()
        config = Config()

    policy = Policy()

    chunk = torch.zeros((1, 3, 7))  # generated chunk_size = 3
    ep = {"task_desc": "t", "max_steps": 6}
    calls = {"predict": 0}

    def fake_predict(_policy, _batch, _noise):
        calls["predict"] += 1
        return chunk, None

    with (patch.object(C, "obs_to_policy", lambda _obs, _desc: {}),
          patch.object(C, "_draw_chunk_noise", lambda *a, **k: None),
          patch.object(C, "predict_clean_chunk", fake_predict),
          patch.object(C, "postprocess_chunk", lambda arr, *a, **k: np.asarray(arr))):
        calls["predict"] = 0
        success, steps = C._run_continuation(
            FakeEnv(), {}, ep, policy, lambda o: o, lambda a: a, torch.device("cpu"),
            prefix=[], branch_seed=1, steps_already=0)
        assert not success and steps == 6 and calls["predict"] == 2  # replan every full chunk (3)
        assert policy.model._pnp.chunk_pos == .5

        calls["predict"] = 0
        positions = []

        def record_position(_policy, _batch, _noise):
            calls["predict"] += 1
            positions.append(_policy.model._pnp.chunk_pos)
            return chunk, None

        with patch.object(C, "predict_clean_chunk", record_position):
            success, steps = C._run_continuation(
                FakeEnv(), {}, ep, policy, lambda o: o, lambda a: a,
                torch.device("cpu"), prefix=[], branch_seed=1,
                steps_already=0, n_action_steps=1)
        assert not success and steps == 6 and calls["predict"] == 6
        assert positions == [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]


def test_sparse_rendering_warms_two_steps_before_each_policy_observation():
    import numpy as np
    import torch
    from unittest.mock import patch
    from pnp.verifier import collection as C

    camera = {"enabled": True}
    rendered_steps = []

    class FakeEnv:
        def step(self, _action):
            rendered_steps.append(camera["enabled"])
            return {}, 0.0, False, {}

        def check_success(self):
            return False

    class PNP:
        chunk_pos = -1.0

    class Model:
        _pnp = PNP()

    class Config:
        chunk_size = 3

    class Policy:
        model = Model()
        config = Config()

    def toggle(_env, enabled):
        camera["enabled"] = bool(enabled)
        return True

    chunk = torch.zeros((1, 3, 7))
    ep = {"task_desc": "t", "max_steps": 6}
    with (patch.object(C, "set_camera_observables", toggle),
          patch.object(C, "obs_to_policy", lambda _obs, _desc: {}),
          patch.object(C, "_draw_chunk_noise", lambda *a, **k: None),
          patch.object(C, "predict_clean_chunk", lambda *a, **k: (chunk, None)),
          patch.object(C, "postprocess_chunk", lambda arr, *a, **k: np.asarray(arr))):
        success, steps = C._run_continuation(
            FakeEnv(), {}, ep, Policy(), lambda o: o, lambda a: a,
            torch.device("cpu"), prefix=[], branch_seed=1, steps_already=0,
            skip_unused_renders=True, render_lead=2)
    assert not success and steps == 6
    assert rendered_steps == [False, True, True, False, True, True]
    assert camera["enabled"] is True


def test_fixed_parent_replay_logs_but_does_not_truncate_terminal_flags():
    import numpy as np
    from unittest.mock import patch
    from pnp.verifier import collection as C

    class FakeEnv:
        def __init__(self):
            self.action_steps = 0

        def reset(self):
            self.action_steps = 0

        def set_init_state(self, _state):
            return {"step": 0}

        def step(self, _action):
            self.action_steps += 1
            return {"step": self.action_steps}, 0.0, self.action_steps == 2, {}

        def check_success(self):
            return self.action_steps == 1

    class FakePolicy:
        def reset(self):
            pass

    env = FakeEnv()
    with patch.object(C, "NUM_STEPS_WAIT", 0):
        obs, events = C._reset_and_replay_actions(
            env, {"init_state": np.zeros(1)}, FakePolicy(),
            np.zeros((3, 7), dtype=np.float32))
    assert env.action_steps == 3
    assert obs == {"step": 3}
    assert events == [
        {"replay_action_index": 0, "success": True, "done": False},
        {"replay_action_index": 1, "success": False, "done": True},
    ]


def test_candidate_collection_uses_persisted_source_state_and_policy_input():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from unittest.mock import patch
    from pnp.verifier import collection as C

    class FakeState:
        def __init__(self, values):
            self.values = np.asarray(values, np.float64).copy()

        def flatten(self):
            return self.values.copy()

    class FakeSim:
        def __init__(self):
            self.values = np.asarray([9.0, 9.0])

        def get_state(self):
            return FakeState(self.values)

        def set_state_from_flattened(self, values):
            self.values = np.asarray(values).copy()

        def set_state(self, state):
            self.values = state.flatten()

        def forward(self):
            pass

    class FakeEnv:
        def __init__(self):
            self.sim = FakeSim()

        def reset(self):
            self.sim.values = np.asarray([9.0, 9.0])

        def set_init_state(self, _state):
            return {"fresh_replay": True}

        def step(self, _action):
            return {"fresh_replay": True}, 0.0, False, {}

        def check_success(self):
            return False

    class Policy:
        config = SimpleNamespace(chunk_size=2, max_action_dim=7)
        model = SimpleNamespace(_pnp=SimpleNamespace(num_steps=10, chunk_pos=-1.0))

        def reset(self):
            pass

    source_observation = {"persisted_source": True}
    seen = []

    def preprocess(observation):
        seen.append(observation)
        return observation

    def predict(_policy, batch, _noise, capture_context=False):
        assert batch is source_observation or batch == source_observation
        return torch.zeros((1, 2, 7)), None

    with (patch.object(C, "NUM_STEPS_WAIT", 0),
          patch.object(C, "_draw_chunk_noise", lambda *a, **k: torch.zeros((1, 2, 7))),
          patch.object(C, "predict_clean_chunk", predict),
          patch.object(C, "postprocess_chunk", lambda value, *a, **k: np.asarray(value)),
          patch.object(C, "_run_continuation", lambda *a, **k: (False, 0))):
        group, _ = C.collect_replay_candidate_group(
            FakeEnv(),
            {"init_state": np.zeros(1), "task_desc": "task", "max_steps": 20,
             "suite": "suite", "task_idx": 0, "ep_idx": 0},
            Policy(), preprocess, lambda value: value, torch.device("cpu"),
            chunk_idx=0, uncertainty_stratum="test", candidate_count=1,
            n_action_steps=2,
            replay_actions_override=np.zeros((0, 7), np.float32),
            source_sim_state_override=np.asarray([3.0, 4.0]),
            source_policy_observation_override=source_observation)

    assert seen == [source_observation]
    assert group["metadata_json"]["canonical_root_source"] == "persisted_source_artifact"
    assert group["metadata_json"]["policy_input_source"] == "persisted_source_artifact"
    assert group["metadata_json"]["persisted_source_state_set_max_abs"] == 0.0
    assert group["metadata_json"]["replay_root_max_abs_vs_canonical"] == 6.0


def test_context_capture_is_noninvasive_and_restores_attention_backend():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from pnp.verifier.collection import predict_clean_chunk

    language_config = SimpleNamespace(_attn_implementation="sdpa")
    language_model = SimpleNamespace(config=language_config)
    paligemma_model = SimpleNamespace(language_model=language_model)
    paligemma = SimpleNamespace(model=paligemma_model)
    expert = SimpleNamespace(paligemma=paligemma)
    model = SimpleNamespace(
        _pnp=SimpleNamespace(strategy=None),
        paligemma_with_expert=expert)

    class FakePolicy:
        def __init__(self):
            self.model = model

        def predict_action_chunk(self, _batch, *, noise):
            tap = self.model._pnp.strategy
            assert tap is not None and not tap.invasive
            language_config._attn_implementation = "eager"
            tap.finish(SimpleNamespace(obs_enc=torch.tensor([[1.0, 2.0]])))
            return noise + 1

    noise = torch.zeros((1, 2, 3))
    chunk, obs_enc = predict_clean_chunk(
        FakePolicy(), {}, noise, capture_context=True)
    assert torch.equal(chunk, torch.ones_like(noise))
    np.testing.assert_array_equal(obs_enc, [[1.0, 2.0]])
    assert language_config._attn_implementation == "sdpa"
    assert model._pnp.strategy is None
