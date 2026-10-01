"""Resume the approved fixed PCP cohort once the original app has drained."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD_APP = "ap-fR5mrDo1v52MGahm5RRi3F"
MODAL = "/Users/arjunsharma/miniconda3/bin/modal"
EXPERIMENT = "smolvla-libero-focused-validation-flow-pcp-v1-9b8800bc9f27116a-d4e3295d4fc3"
SHA = "9b8800bc9f27116afcb1812f93f8a9f6767ac012c7cdac790502032fbb7d0c8e"


def complete_roots():
    from scripts.train_smolvla_combined_local import _load_local_credentials
    from pnp.store import SupabaseStore
    from pnp.smolvla_pcp_pilot import complete_candidate_contract
    _load_local_credentials()
    store = SupabaseStore()
    groups = store.fetch_all("verifier_candidate_groups", "candidate_group_id",
        configure=lambda q: q.eq("experiment", EXPERIMENT), order_by=("candidate_group_id",))
    count = 0
    for group in groups:
        candidates = store.fetch_all("verifier_candidates", "candidate_kind,metadata_json,success,n_steps",
            configure=lambda q: q.eq("candidate_group_id", group["candidate_group_id"]),
            order_by=("candidate_kind",))
        count += complete_candidate_contract(candidates, expected_critic_sha=SHA)
    return count


def main():
    diagnostics = Path.home() / "pnp-vla-runs/diagnostics"
    lock = (diagnostics / "modal_pcp_completion_resume.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("PCP completion supervisor already running")
    deadline = time.monotonic() + 60 * 60
    previous = None
    while True:
        result = subprocess.run([MODAL, "container", "list", "--app-id", OLD_APP, "--json"],
                                capture_output=True, text=True, check=True, timeout=60)
        containers = json.loads(result.stdout)
        if len(containers) != previous:
            print({"phase": "waiting_for_original_workers", "active": len(containers),
                   "original_app": OLD_APP}, flush=True)
            previous = len(containers)
        if not containers:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Old app has not drained; refuse overlapping resume workers")
        time.sleep(30)
    # Ensure the old ephemeral app cannot schedule more work after the zero check.
    subprocess.run([MODAL, "app", "stop", OLD_APP], check=True, timeout=60)
    count = complete_roots()
    print({"phase": "completion_inventory", "complete_roots": count, "planned_roots": 42}, flush=True)
    if count == 42:
        print({"phase": "already_complete"}, flush=True)
        return
    if not 0 <= count < 42:
        raise ValueError("Unexpected root count; fixed cohort differs")
    print({"phase": "launching_approved_resume", "worker_timeout_minutes": 90,
           "pending_roots": 42 - count, "max_concurrent_workers": 3}, flush=True)
    # User explicitly approved the longer completion pass. The collector skips
    # groups with all nine complete artifacts and the fixed checkpoint contract.
    subprocess.run([MODAL, "run", "scripts/modal_smolvla_flow_pcp.py", "--no-smoke", "--approve-spend"],
                   cwd=ROOT, check=True)
    count = complete_roots()
    print({"phase": "resume_finished", "complete_roots": count, "planned_roots": 42}, flush=True)
    if count != 42:
        raise RuntimeError("Resume returned without a complete fixed cohort")


if __name__ == "__main__":
    main()
