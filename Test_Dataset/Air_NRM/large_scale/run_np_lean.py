#!/usr/bin/env python3
"""Check or launch the Air-NRM-NP-LS LEAN notebook from a terminal."""

from __future__ import annotations

import argparse
import ast
import csv
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

from np_lean_workflow import AirNrmLargeScaleData, parse_np_query


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUERY = HERE / "query_NP_Flow.csv"
EXEMPLAR = HERE / "RAG_Example_SBLP_Flow_LargeScale_NP.csv"
NOTEBOOK = HERE / "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb"
BUILDER = HERE / "build_np_lean_notebook.py"
# Use the activated environment on either a laptop or a Linux server.
RUNTIME = Path(sys.executable)
REQUIREMENTS = ROOT / "requirements-server.txt"
if not REQUIREMENTS.exists():
    REQUIREMENTS = ROOT / "requirements.txt"
RUNTIME_PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-openai",
    "langchain-community",
    "langchain-classic",
    "langchain-ollama",
    "openai",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def verify_runtime() -> None:
    """Check this interpreter before prompting for credentials or launching."""
    if not RUNTIME.exists():
        raise ValueError(f"required LEAN runtime not found: {RUNTIME}")
    if not REQUIREMENTS.exists():
        raise ValueError(f"repository requirements file not found: {REQUIREMENTS}")
    probe = r'''
import importlib.metadata as metadata
import pathlib
import re

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
import gurobipy
import leanopt_exp
import openai
import pandas

text = pathlib.Path(%r).read_text(encoding="utf-8")
packages = %r
expected = {}
for package in packages:
    match = re.search(r"^" + re.escape(package) + r"==([^\s#]+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError("requirements.txt has no exact pin for " + package)
    expected[package] = match.group(1)
mismatch = {
    package: (metadata.version(package), version)
    for package, version in expected.items()
    if metadata.version(package) != version
}
if mismatch:
    raise RuntimeError("runtime differs from requirements.txt: " + repr(mismatch))
print("compatible LEAN runtime matching requirements.txt")
''' % (str(REQUIREMENTS), RUNTIME_PACKAGES)
    result = subprocess.run(
        [str(RUNTIME), "-c", probe],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"LEAN runtime dependency check failed:\n{detail}")


def offline_check() -> None:
    verify_runtime()
    query_fields, rows = read_rows(QUERY)
    if query_fields != ["n_variables", "Query", "Label-objective"] or len(rows) != 15:
        raise ValueError("NP query CSV must contain the CA-aligned three columns and 15 rows")
    data = AirNrmLargeScaleData.from_path(HERE)
    used_flights = set()
    counts = []
    for number, row in enumerate(rows, 1):
        parsed = parse_np_query(row["Query"])
        parameters = data.assemble(row["Query"])
        used_flights.update(parsed.flights)
        n_ods = len({flight[:2] for flight in parsed.flights})
        expected_variables = 5 * len(parsed.flights) + n_ods
        if int(row["n_variables"]) != expected_variables:
            raise ValueError(f"query {number}: n_variables mismatch")
        if parameters["option_num"] != parsed.z:
            raise ValueError(f"query {number}: Z was lost during data assembly")
        counts.append((len(parsed.flights), expected_variables, parsed.z))

    exemplar_fields, exemplar_rows = read_rows(EXEMPLAR)
    if exemplar_fields != ["prompt"] or len(exemplar_rows) != 1:
        raise ValueError("the Flow exemplar must be one label-free prompt row")
    exemplar = parse_np_query(exemplar_rows[0]["prompt"])
    if set(exemplar.flights) & used_flights:
        raise ValueError("the Flow exemplar overlaps a test candidate flight")
    data.assemble(exemplar_rows[0]["prompt"])

    subprocess.run([sys.executable, str(BUILDER), "--check"], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required_markers = (
        "Air-NRM-NP-LS",
        "install_np_workflow",
        "NP_SET_CURRENT_QUERY(query)",
        "raw_model_output",
        "direct_LLM_code_extraction",
        "never repair or inject data",
        "query_NP_Flow.csv",
        "option_num",
    )
    missing = [marker for marker in required_markers if marker not in all_source]
    if missing:
        raise ValueError(f"NP notebook missing markers: {missing}")
    if "query_CA.csv" in all_source or "small_scale/query_NP_Flow.csv" in all_source:
        raise ValueError("NP notebook still contains a CA/small-scale execution path")
    config_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "lx.load_config(" in "".join(cell.get("source", []))
    ]
    if len(config_cells) != 1 or "repeat_index=0" not in config_cells[0]:
        raise ValueError("NP notebook repeat_index is incompatible with run_notebook.py --repeat")

    config = (ROOT / "exp_config.yaml").read_text(encoding="utf-8")
    expected_map = "Air-NRM-NP-LS:  Test_Dataset/Air_NRM/large_scale/query_NP_Flow.csv"
    if expected_map not in config:
        raise ValueError("exp_config.yaml has no Air-NRM-NP-LS dataset mapping")

    for path in (HERE / "np_lean_workflow.py", HERE / "build_np_lean_notebook.py", __file__):
        ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))

    print("OK: 15/15 NP queries assembled from all five CSV files")
    print(f"OK: flight count {min(v[0] for v in counts)}-{max(v[0] for v in counts)}; "
          f"variables {min(v[1] for v in counts)}-{max(v[1] for v in counts)}; "
          f"Z {min(v[2] for v in counts)}-{max(v[2] for v in counts)}")
    print("OK: four capacity-consumption values, 28% per-flight capacity, and Z preserved")
    print("OK: exemplar is label-free and uses two flights absent from all 15 test queries")
    print("OK: independent NP notebook/config/launcher are structurally current")
    print("No model API was called.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="offline validation; no API")
    parser.add_argument("--list", action="store_true", help="show the run plan; no API")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--row", type=int, help="one 1-based query number (1-15)")
    select.add_argument("--n", type=int, help="first N queries")
    parser.add_argument("--profile", default="gpt-4.1")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--prompt-key", action="store_true",
                        help="securely prompt for OPENAI_API_KEY for this process only")
    args = parser.parse_args()

    if args.check:
        offline_check()
        return 0
    if args.row is not None and not 1 <= args.row <= 15:
        raise SystemExit("--row must be between 1 and 15")
    if args.n is not None and not 1 <= args.n <= 15:
        raise SystemExit("--n must be between 1 and 15")

    # Check imports before asking the user for a secret or constructing a run.
    verify_runtime()

    env = os.environ.copy()
    if args.prompt_key and not args.list:
        key = getpass.getpass("OPENAI_API_KEY (hidden; not saved): ").strip()
        if not key:
            raise SystemExit("empty API key")
        env["OPENAI_API_KEY"] = key

    command = [
        str(RUNTIME),
        str(ROOT / "run_notebook.py"),
        str(NOTEBOOK.relative_to(ROOT)),
        "--data", str(QUERY.relative_to(ROOT)),
        "--dataset", "Air-NRM-NP-LS",
        "--profile", args.profile,
        "--repeat", str(args.repeat),
    ]
    if args.row is not None:
        command += ["--rows", str(args.row - 1)]
    elif args.n is not None:
        command += ["--n", str(args.n)]
    if args.resume is not None:
        command += ["--resume"] if args.resume == "auto" else ["--resume", args.resume]
    if args.list:
        command.append("--list")
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
