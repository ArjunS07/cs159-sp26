"""Resumption guards: a restarted worker must see every already-logged rollout.

existing_keys is the whole basis of "reopen the worker and Run all". If it silently truncates,
completed rollouts look pending and get recollected -- idempotent, but hours of wasted GPU.
"""
from types import SimpleNamespace

from pnp.store import SupabaseStore


class Query:
    """Minimal PostgREST stand-in that enforces a hard per-request row cap."""

    def __init__(self, rows, cap):
        self.rows, self.cap = rows, cap
        self.filters = {}
        self.page = []
        self.ordered = []

    def select(self, *_):
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def order(self, column):
        self.ordered.append(column)
        return self

    def range(self, start, end):
        matching = [
            row for row in self.rows
            if all(row.get(k) == v for k, v in self.filters.items())
        ]
        requested = end - start + 1
        self.page = matching[start:start + min(requested, self.cap)]
        return self

    def execute(self):
        return SimpleNamespace(data=self.page)


class Client:
    def __init__(self, rows, cap=1000):
        self.rows, self.cap = rows, cap
        self.tables = []

    def table(self, name):
        self.tables.append(name)
        return Query(self.rows, self.cap)


def _store(rows, cap=1000):
    store = object.__new__(SupabaseStore)
    store.client = Client(rows, cap)
    return store


def test_existing_keys_pages_past_the_row_cap():
    rows = [{"rollout_id": f"r{i:05d}", "experiment": "exp"} for i in range(2500)]
    keys = _store(rows).existing_keys("exp")
    assert len(keys) == 2500
    assert keys == {row["rollout_id"] for row in rows}


def test_existing_keys_filters_by_experiment_across_pages():
    rows = ([{"rollout_id": f"a{i:05d}", "experiment": "mine"} for i in range(1500)]
            + [{"rollout_id": f"b{i:05d}", "experiment": "other"} for i in range(1500)])
    keys = _store(rows).existing_keys("mine")
    assert len(keys) == 1500
    assert all(key.startswith("a") for key in keys)


def test_existing_keys_honours_extra_equality_filters():
    rows = [{"rollout_id": f"r{i:05d}", "experiment": "exp",
             "suite": "libero_goal_swap" if i % 2 else "libero_10_swap"}
            for i in range(1200)]
    keys = _store(rows).existing_keys("exp", suite="libero_goal_swap")
    assert len(keys) == 600


def test_existing_keys_orders_for_deterministic_pagination():
    """Unordered pagination can repeat or skip rows between requests, so every request must
    order by rollout_id. One shared Query stands in for all of them, so it accumulates one
    entry per page."""
    rows = [{"rollout_id": f"r{i:05d}", "experiment": "exp"} for i in range(1200)]
    store = _store(rows)
    query = Query(rows, 1000)
    store.client.table = lambda name: query
    store.existing_keys("exp")
    assert query.ordered  # at least one page requested
    assert set(query.ordered) == {"rollout_id"}


def test_iter_todo_skips_completed_and_yields_the_rest():
    from pnp.config import Method, RolloutConfig

    episodes = [{"benchmark": "libero_pro", "suite": "s", "task_idx": 0, "ep_idx": i,
                 "init_state_hash": f"h{i}"} for i in range(3)]
    methods = [(Method.UNCERTAINTY, RolloutConfig(pnp_steps=(3, 4), pnp_k=5)),
               (Method.REFINEMENT, RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True))]
    store = _store([])
    all_ids = {store.rollout_id("exp", ep, name, cfg)
               for ep in episodes for name, cfg in methods}
    assert len(all_ids) == 6

    done = {store.rollout_id("exp", episodes[0], *methods[0])}
    todo = list(store.iter_todo("exp", episodes, methods, done=done))
    assert len(todo) == 5
    assert done.isdisjoint({rid for *_, rid in todo})


def test_multisample_denorm_records_its_probe_and_candidate_identity():
    from pnp.config import Method, RolloutConfig

    config = RolloutConfig(
        num_samples=2, pnp_k=5, ms_probe_steps=(3, 4),
        candidate_set_id="source@a|m1@b")
    row = SupabaseStore._denorm(Method.CHUNK_SOURCE_M1, config)
    assert row["pnp_enabled"] is True
    assert row["pnp_k"] == 5
    assert row["pnp_step_indices"] == [3, 4]
    assert config.logical_dict()["candidate_set_id"] == "source@a|m1@b"
    assert "candidate_set_id" not in RolloutConfig().logical_dict()
