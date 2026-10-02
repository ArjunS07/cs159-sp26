"""Bounded, resumable overnight MC/TD comparisons; one Metal worker at a time."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path.home() / "pnp-vla-runs"
DIGEST = "671b5b211099997fc83d1277"
BASE = ROOT / "checkpoints/smolvla-q10-scalar-mc-td-v1-preaction" / DIGEST
OUTPUT = ROOT / "diagnostics/overnight_20260930"
JOBS = [
    ("mc_roots", "mc", ["--root-fraction", ".5"]),
    ("td_n5", "td", ["--td-steps", "5"]),
    ("mc_long", "mc", []),
    ("td_long", "td", []),
    ("mc_roots_late", "mc", ["--root-fraction", ".5", "--architecture", "late_fusion"]),
    ("mc_roots_contextdrop", "mc", ["--root-fraction", ".5", "--prefix-dropout", ".1"]),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    deadline = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # 8am Pacific; finish current job.
    rows = []
    for label, arm, options in JOBS:
        name = "smolvla-overnight-20260930-" + label
        report = ROOT / "checkpoints" / name / DIGEST / arm / "report.json"
        old = json.loads(report.read_text()) if report.exists() else None
        final_checkpoint = report.parent / "checkpoint_step_004000.pt"
        complete = (old is not None and old["history"][-1]["update"] == 4000
                    and final_checkpoint.exists()
                    and torch.load(final_checkpoint, map_location="cpu", weights_only=False)["update"] == 4000)
        if not complete and datetime.now(timezone.utc) >= deadline:
            print({"status": "morning boundary reached; no new jobs started", "remaining": label}, flush=True)
            break
        command = [sys.executable, "-u", str(REPO / "scripts/train_smolvla_scalar_returns_local.py"),
                   "--experiment-name", name, "--arms", arm, "--init-from", str(BASE / arm / "latest.pt"),
                   "--reset-optimizer", "--updates", "4000", "--lr", "0.00005", *options]
        code = 0
        if not complete:
            print({"starting": label, "objective": arm, "options": options}, flush=True)
            with (OUTPUT / (label + ".log")).open("a") as log:
                code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        rows.append({"job": label, "status": "finished" if code == 0 else "failed",
                     "exit_code": code, "report": str(report)})
        temporary = OUTPUT / "status.tmp"
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(OUTPUT / "status.json")
        if code:
            print({"job_failed": label, "exit_code": code}, flush=True)
            return code
        diagnostic = OUTPUT / (label + "_diagnostics.json")
        if not diagnostic.exists():
            with (OUTPUT / (label + "_diagnostics.log")).open("a") as log:
                subprocess.run([sys.executable, "-u", str(REPO / "scripts/diagnose_smolvla_scalar_returns.py"),
                                "--checkpoint", str(final_checkpoint), "--output", str(diagnostic)],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        print({"finished": label, "report": str(report), "diagnostics": str(diagnostic)}, flush=True)
    print({"overnight_status": rows}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
