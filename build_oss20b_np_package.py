#!/usr/bin/env python3
"""Build a fresh, allowlisted, self-contained Air-NRM-NP-LS server package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "oss20b_np_pkg_src"
DATA = Path("Test_Dataset/Air_NRM/large_scale")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory; existing directories are never overwritten")
    args = parser.parse_args()
    output = (args.output or ROOT / "deliverables" /
              ("airnrm_np_oss20b_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}; choose a fresh path")
    subprocess.run([sys.executable, str(ROOT / DATA / "build_np_lean_notebook.py"), "--check"],
                   cwd=ROOT, check=True)
    names = ["leanopt_exp.py", "exp_config.yaml", "run_notebook.py", "score_runs.py",
             "aggregate_runs.py", "Large_Scale_Or_Files/RefData.csv"]
    names += [str(DATA / name) for name in (
        "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb",
        "LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb",  # source for reproducibility check
        "np_lean_workflow.py", "run_np_lean.py", "build_np_lean_notebook.py",
        "query_NP_Flow.csv", "v1.csv", "v2.csv", "od_demand.csv", "flight.csv",
        "flight_capacity.csv", "RAG_Example_SBLP_Flow_LargeScale_NP.csv",
        "SBLP_NP_Flow_Label/np_obj.csv",
    )]
    # An explicit allowlist excludes .env, credentials, old results, venvs and caches.
    output.mkdir(parents=True)
    for name in names:
        source, destination = ROOT / name, output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.suffix == ".ipynb":
            notebook = json.loads(destination.read_text(encoding="utf-8"))
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
            destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("run_airnrm_np.py", "check_server.py", "verify_package.py", "setup.sh", "requirements-server.txt", "README.md"):
        shutil.copyfile(SOURCE / name, output / name)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "Air-NRM-NP-LS", "profile": "gpt-oss-20b", "sha256": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["sha256"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(str(output) + ".tar.gz", "w:gz") as tar:
        tar.add(output, arcname=output.name)
    with zipfile.ZipFile(str(output) + ".zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(output.name) / path.relative_to(output)))
    print(f"Folder: {output}\nLinux:  {output}.tar.gz\nZip:    {output}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
