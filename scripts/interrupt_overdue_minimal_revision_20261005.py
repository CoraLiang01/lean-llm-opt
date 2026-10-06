"""Apply the same external deadline to workers started before its adoption.

Only signal a child of this campaign's verified coordinator with an open case log.
Never infer completion from an incumbent; an interrupted solver is scored as failed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / "outputs/minimal_revision_20261005"


def check(version):
    root = ROOT / ("full_" + version)
    coordinator = int((root / "coordinator_pid.txt").read_text())
    processes = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    children = []
    verified = False
    for row in processes.splitlines():
        pid, parent, command = row.strip().split(None, 2)
        if int(pid) == coordinator:
            verified = "evaluate_minimal_revision_20261005.py" in command and f"--version {version}" in command
        if int(parent) == coordinator and "multiprocessing.spawn" in command:
            children.append(int(pid))
    if not verified:
        return False
    for pid in children:
        output = subprocess.run(["lsof", "-p", str(pid), "-Fn"], capture_output=True, text=True).stdout
        for line in output.splitlines():
            if not line.startswith("n"):
                continue
            log = Path(line[1:])
            if log.name != "run.log" or not log.is_relative_to(root):
                continue
            folder = log.parent
            marker = folder / "attempt.json"
            if not marker.exists() or (folder / "result.json").exists():
                continue
            elapsed = time.time() - marker.stat().st_mtime
            control = folder / "external_interrupt.json"
            if elapsed >= 1800 and not control.exists():
                event = {"utc": datetime.now(timezone.utc).isoformat(), "pid": pid,
                         "elapsed_seconds": round(elapsed, 3), "deadline_seconds": 1800,
                         "reason": "Common external case deadline exceeded", "signal": "SIGINT",
                         "rerun": False, "score": "Failure unless an optimal result was already completed"}
                control.write_text(json.dumps(event, indent=2))
                os.kill(pid, signal.SIGINT)
                print(json.dumps({"version": version, "case": folder.name, **event}), flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v3", "v4"])
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        active = [check(version) for version in args.versions]
        if not args.watch or not any(active):
            break
        time.sleep(10)
