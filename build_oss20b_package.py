#!/usr/bin/env python3
"""
build_oss20b_package.py
=======================
Assemble `oss20b_run/` -- the self-contained folder that goes to the coauthor
whose server runs the gpt-oss-20B experiments.

    python build_oss20b_package.py
    python build_oss20b_package.py --tar        # also write oss20b_run.tar.gz

Why a script instead of copying files by hand
---------------------------------------------
The smoke test will find something, we will fix it in the repository, and the
package has to be rebuilt. A hand-copied package drifts from the repository
after the first such round -- and the drifted copy is the one that was sent.
Running this again is the whole update procedure.

The build is idempotent: `oss20b_run/` is deleted and rebuilt each time.
Nothing in the repository is modified; the two config edits that the server
needs (num_ctx, keep_alive, reasoning, the OLLAMA_BASE_URL template) already
live in the repository's own exp_config.yaml and are copied verbatim.

Layout produced
---------------
    oss20b_run/
      README.md  requirements-server.txt  .env.example  verify_package.py
      package_manifest.json
      large_scale_or/   notebook + code + data + runners + scoring
      loto/             LOTO notebook + LOTO code + data + runners

Both task directories carry their own copy of the data (~5 MB each) rather
than sharing one through a symlink: `LOTO/Test_Dataset` in the repository is a
symlink to `../Test_Dataset`, and symlinks do not survive every zip/unzip
round trip. When one breaks, the error is "file not found" three cells into a
run, which reads as a data problem rather than a packaging problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "oss20b_pkg_src"
OUT = HERE / "oss20b_run"

# The LOTO notebook is renamed on the way in. Two files with the same name and
# different contents, one folder apart, is an invitation to copy one over the
# other -- and a LOTO run driven by the unpatched notebook produces numbers
# that look entirely normal while measuring nothing.
LOTO_NOTEBOOK_SRC = "LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb"
LOTO_NOTEBOOK_DST = "LEAN_LLM_OPT_gpt_oss_20b_LOTO.ipynb"

# ---- what each task directory gets ---------------------------------------- #
# (source path relative to the repo root, destination name)
MAIN_FILES = [
    ("LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb", None),
    ("leanopt_exp.py", None),
    ("exp_config.yaml", None),
    ("run_notebook.py", None),
    ("set_key.py", None),
    ("score_runs.py", None),
    ("aggregate_runs.py", None),
]

LOTO_FILES = [
    (f"LOTO/{LOTO_NOTEBOOK_SRC}", LOTO_NOTEBOOK_DST),
    ("LOTO/leanopt_exp.py", None),
    ("LOTO/exp_config.yaml", None),
    ("LOTO/run_loto_fold.py", None),
    ("LOTO/loto_report.py", None),
    ("LOTO/run_notebook.py", None),
    ("set_key.py", None),
    ("score_runs.py", None),
]

AIRNRM_FILES = [
    ("LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb", None),
    ("leanopt_exp.py", None),
    ("exp_config.yaml", None),
    ("run_notebook.py", None),
    ("set_key.py", None),
    ("score_runs.py", None),
    ("aggregate_runs.py", None),
]

# Data each task needs, as (source, destination) pairs.
#
# Air-NRM ships large_scale only. The forked notebook reads its five tables --
# v1, v2, od_demand, flight and flight_capacity -- from that directory and
# names no small-scale file in any cell that executes; the small-scale paths
# that remain are inside the commented-out NP-Flow cells, which this dataset
# has no query set for. Large_Scale_Or_Files comes along for RefData.csv
# (classification) and the SBLP few-shot corpora.
DATA_DIRS = {
    "large_scale_or": [
        ("Large_Scale_Or_Files", "Large_Scale_Or_Files"),
        ("Test_Dataset/Large-scale-or", "Test_Dataset/Large-scale-or"),
    ],
    "loto": [
        ("Large_Scale_Or_Files", "Large_Scale_Or_Files"),
        ("Test_Dataset/Large-scale-or", "Test_Dataset/Large-scale-or"),
    ],
    "air_nrm_ls": [
        ("Large_Scale_Or_Files", "Large_Scale_Or_Files"),
        ("Test_Dataset/Air_NRM/large_scale", "Test_Dataset/Air_NRM/large_scale"),
    ],
}

TASKS = ["large_scale_or", "loto", "air_nrm_ls"]

# Never ship these, whatever directory they turn up in. `.env` is the one that
# matters: LOTO/ currently holds two of them.
EXCLUDE_NAMES = {".DS_Store", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_PREFIXES = (".env",)

# Only actual data comes out of the data trees. Those directories also hold
# development leftovers -- old notebooks, one-off solver scripts, a .docx --
# and one of the notebooks, Large_Scale_Or_Files/Other_example/
# LEAN_LLM_OPT_Large-scale-or.ipynb, has a hard-coded OpenAI key in it. The
# audit at the end of this script caught it; this allowlist is what stops it
# from being there in the first place. Every one of the 210 files the pipeline
# actually reads is a .csv, so nothing needed is lost -- verify_package.py
# re-resolves all 210 in the built package and fails if any went missing.
DATA_SUFFIXES = {".csv", ".txt", ".lp", ".xlsx"}

SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".txt",
                 ".ipynb", ".csv", ".example"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skip(path: Path) -> bool:
    return (path.name in EXCLUDE_NAMES
            or path.name.startswith(EXCLUDE_PREFIXES)
            or any(part in EXCLUDE_NAMES for part in path.parts))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> int:
    n = skipped = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file() or skip(path):
            continue
        if path.suffix.lower() not in DATA_SUFFIXES:
            skipped += 1
            continue
        copy_file(path, dst / path.relative_to(src))
        n += 1
    if skipped:
        print(f"    ({skipped} non-data file(s) left behind)")
    return n


def build_task(name: str, files: list[tuple[str, str | None]]) -> Path:
    task = OUT / name
    task.mkdir(parents=True, exist_ok=True)

    for rel, newname in files:
        src = HERE / rel
        if not src.exists():
            sys.exit(f"missing source file: {rel}")
        copy_file(src, task / (newname or Path(rel).name))

    for src_rel, dst_rel in DATA_DIRS[name]:
        n = copy_tree(HERE / src_rel, task / dst_rel)
        print(f"    {dst_rel}: {n} file(s)")

    # Optional per-task extras, if oss20b_pkg_src/<task>/ exists. The runners
    # themselves are shared (run_oss20b.py at the package root), so this is
    # usually empty.
    for extra in sorted((SRC / name).glob("*")) if (SRC / name).is_dir() else []:
        dst = task / extra.name
        copy_file(extra, dst)
        if extra.suffix == ".sh":
            dst.chmod(0o755)

    (task / "runs").mkdir(exist_ok=True)
    (task / "runs" / ".gitkeep").write_text("", encoding="utf-8")
    return task


def patch_loto_runner(task: Path) -> None:
    """Point the packaged runner at the packaged (renamed) notebook."""
    p = task / "run_loto_fold.py"
    s = p.read_text(encoding="utf-8")
    old = 'DEFAULT_NOTEBOOK = "LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"'
    if old not in s:
        sys.exit("run_loto_fold.py no longer has the DEFAULT_NOTEBOOK line "
                 "this script patches; update build_oss20b_package.py")
    new = (f'# Packaged build: the 4.1 notebook is not shipped here, and the\n'
           f'# gpt-oss LOTO notebook is renamed so it cannot be confused with\n'
           f'# the main-experiment one in ../large_scale_or/.\n'
           f'DEFAULT_NOTEBOOK = "{LOTO_NOTEBOOK_DST}"')
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print("    run_loto_fold.py: DEFAULT_NOTEBOOK -> " + LOTO_NOTEBOOK_DST)


def write_manifest() -> dict:
    """Hash every packaged file so the receiver can prove nothing rotted."""
    files: dict[str, dict[str, str]] = {}
    for task in TASKS:
        entries = {}
        root = OUT / task
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "runs" in path.relative_to(root).parts:
                continue
            entries[str(path.relative_to(root))] = sha256_file(path)
        files[task] = entries

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=HERE, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except Exception:                                             # noqa: BLE001
        commit = None

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": commit,
        "files": files,
    }
    (OUT / "package_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit() -> int:
    """Last gate before this leaves the machine."""
    problems = []

    # __pycache__ is excluded during the copy, but verify_package.py imports
    # leanopt_exp from inside a task directory, so running the self-check
    # creates one *after* the build. Catch it by path component, not by file
    # name -- the offending files are called *.pyc, not __pycache__.
    strays = [p for p in OUT.rglob("*")
              if p.is_file() and (p.name.startswith(".env")
                                  and p.name != ".env.example"
                                  or p.name in EXCLUDE_NAMES
                                  or any(part in EXCLUDE_NAMES for part in p.parts))]
    for p in strays:
        problems.append(f"stray file shipped: {p.relative_to(OUT)}")

    for path in OUT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                         # noqa: BLE001
            continue
        if SECRET_RE.search(text):
            problems.append(f"key-shaped string in {path.relative_to(OUT)}")

    # The two config edits the server depends on. Getting here with either one
    # missing means the package would run and quietly produce bad numbers.
    for task in TASKS:
        cfg = (OUT / task / "exp_config.yaml").read_text(encoding="utf-8")
        if "num_ctx" not in cfg:
            problems.append(f"{task}/exp_config.yaml has no num_ctx")
        if "OLLAMA_BASE_URL" not in cfg:
            problems.append(f"{task}/exp_config.yaml has no OLLAMA_BASE_URL "
                            f"template")

    for p in problems:
        print(f"  [FAIL] {p}")
    return len(problems)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="store_true",
                    help="also write oss20b_run.tar.gz next to the folder")
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"package sources not found: {SRC}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    print("shared files")
    for name in ("README.md", "requirements-server.txt", ".env.example",
                 "verify_package.py", "check_smoke.py", "run_oss20b.py",
                 "dry_run_setup.py",
                 "setup_and_smoke.sh"):
        dst = OUT / name
        copy_file(SRC / name, dst)
        if dst.suffix == ".sh":
            dst.chmod(0o755)
        print(f"    {name}")

    print("large_scale_or/")
    build_task("large_scale_or", MAIN_FILES)

    print("loto/")
    task = build_task("loto", LOTO_FILES)
    patch_loto_runner(task)

    print("air_nrm_ls/")
    build_task("air_nrm_ls", AIRNRM_FILES)

    print("manifest")
    manifest = write_manifest()
    n_files = sum(len(v) for v in manifest["files"].values())
    print(f"    {n_files} file(s) hashed")

    print("audit")
    n_bad = audit()
    if n_bad:
        print(f"\n{n_bad} problem(s) -- package NOT safe to send")
        return 1
    print("    clean: no .env, no key-shaped strings, config patches present")

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\nbuilt {OUT.relative_to(HERE)}/  ({size / 1e6:.1f} MB, "
          f"{n_files} files)")

    if args.tar:
        tar = HERE / "oss20b_run.tar.gz"
        subprocess.run(["tar", "czf", str(tar), "-C", str(HERE), OUT.name],
                       check=True)
        print(f"wrote {tar.name} ({tar.stat().st_size / 1e6:.1f} MB)")

    print("\nnext:")
    print("  cd oss20b_run/large_scale_or && python ../verify_package.py")
    print("  cd oss20b_run/loto           && python ../verify_package.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
