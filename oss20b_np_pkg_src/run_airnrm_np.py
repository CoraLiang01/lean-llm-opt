#!/usr/bin/env python3
"""Portable Air-NRM-NP-LS runner: local/remote Ollama, smoke, resume, return bundle."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("Test_Dataset/Air_NRM/large_scale")
NOTEBOOK = DATA / "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb"
PROFILE = "gpt-oss-20b"


class MissingModelsError(RuntimeError):
    """The endpoint is Ollama, but its advertised model tags are incomplete."""


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_command(command: list[str], env: dict | None = None) -> int:
    print("$ " + shlex.join(command), flush=True)
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
        return child.wait()


def api(base: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        hint = (
            "This package needs Ollama's /api/* protocol; a vLLM/SGLang "
            "OpenAI-compatible /v1 endpoint needs a different adapter."
            if exc.code in (404, 405) else
            "Check server permissions/authentication or its service logs."
        )
        raise RuntimeError(f"HTTP {exc.code} from {base}{path}. {hint}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Ollama request failed: {base}{path}: {exc}. "
            "Start Ollama on the inference server, or pass "
            "--ollama-url http://SERVER:11434 (or set OLLAMA_BASE_URL)."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"{base}{path} did not return valid JSON. Check the URL; it may point "
            "to a web page/proxy instead of the Ollama API."
        ) from exc


def normalize_ollama_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must be http(s)://HOST:PORT, without query or fragment")
    if parsed.path.endswith("/v1"):
        raise ValueError(
            "The URL ends with /v1 (OpenAI-compatible interface). This package "
            "requires an Ollama base URL such as http://HOST:11434. "
            "If the service is vLLM/SGLang, its adapter must be configured separately."
        )
    # Accept either the service root or the /api root; never request /api/api/tags.
    return value[:-4] if parsed.path.endswith("/api") else value


def preflight(base: str) -> dict:
    base = normalize_ollama_url(base)
    tags = api(base, "/api/tags")
    if not isinstance(tags, dict) or not isinstance(tags.get("models"), list):
        raise RuntimeError(
            f"{base}/api/tags does not have Ollama's models array. "
            "Check the server API type and any reverse proxy."
        )
    available = {item.get("name") or item.get("model") for item in tags["models"]
                 if isinstance(item, dict) and isinstance(item.get("name") or item.get("model"), str)}
    wanted = ("gpt-oss:20b", "nomic-embed-text")
    missing = [name for name in wanted
               if name not in available and name + ":latest" not in available]
    if missing:
        raise MissingModelsError(
            f"Ollama reachable at {base}, but required models are missing: "
            + ", ".join(missing) + ". Available tags: "
            + (", ".join(sorted(available)) or "(none)") + ". On the server run: "
            + "; ".join("ollama pull " + name for name in missing)
        )
    info = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_host": platform.node(), "runner_platform": platform.platform(),
            "python": sys.version, "ollama_url": base,
            "available_tags": sorted(available), "models": {}}
    # Some proxies expose inference/tags but not diagnostics. Missing hardware
    # metadata is recorded without preventing an otherwise valid experiment.
    for label, path, payload in [("ollama_version", "/api/version", None),
                                 *[(name, "/api/show", {"model": name}) for name in wanted]]:
        try:
            value = api(base, path, payload)
        except RuntimeError as exc:
            value = {"unavailable": str(exc)}
            print(f"Metadata unavailable ({label}); continuing: {exc}")
        if label == "ollama_version":
            info[label] = value
        else:
            info["models"][label] = value
    # This GPU belongs to the Python runner host. A remote Ollama host may differ.
    gpu = shutil.which("nvidia-smi")
    if gpu:
        try:
            result = subprocess.run(
                [gpu, "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                capture_output=True, text=True, timeout=15, check=False)
            info["runner_host_gpu"] = (result.stdout or result.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["runner_host_gpu"] = f"unavailable: {exc}"
    else:
        info["runner_host_gpu"] = "nvidia-smi not found on runner host"
    info["hardware_note"] = os.environ.get("HARDWARE_NOTE", "")
    print("Ollama reachable; exact gpt-oss:20b and nomic-embed-text tags present.")
    return info


def query_rows() -> list[dict]:
    with (ROOT / DATA / "query_NP_Flow.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def smoke_ids() -> list[int]:
    sizes = [int(row["n_variables"]) for row in query_rows()]
    return sorted({min(range(len(sizes)), key=sizes.__getitem__),
                   max(range(len(sizes)), key=sizes.__getitem__)})


def latest_run(output: Path, repeat: int) -> Path | None:
    prefix = f"LEAN-LLM-OPT__{PROFILE}__Air-NRM-NP-LS__r{repeat}__"
    hits = [p for p in output.glob(prefix + "*") if (p / "instances.jsonl").exists()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def validate_run(directory: Path | None, expected: list[int]) -> bool:
    if directory is None:
        print("FAIL: no run directory/instances.jsonl was produced.")
        return False
    from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
        AirNrmLargeScaleData, validate_complete_generated_answer,
    )
    data = AirNrmLargeScaleData.from_path(ROOT / DATA)
    queries = query_rows()
    records = {}
    for line in (directory / "instances.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # A resume may append a later successful attempt for the same query.
        records[int(record["instance_id"])] = record
    failures = []
    for idx in expected:
        record = records.get(idx, {})
        try:
            if record.get("status") != "ok" or record.get("pred_type") != "Policy_Flow":
                raise ValueError(str(record.get("errors") or "missing/failed NP record"))
            extracted = validate_complete_generated_answer(
                record.get("model_output", ""), data.assemble(queries[idx]["Query"]))
            if extracted != record.get("code_output"):
                raise ValueError("code_output differs from the original model answer")
        except (ValueError, TypeError, SyntaxError) as exc:
            failures.append(idx + 1)
            print(f"FAIL query {idx + 1}: {exc}")
    print(f"NP output checks: {len(expected) - len(failures)}/{len(expected)} passed. "
          "This checks complete raw outputs, not solver accuracy.")
    return not failures


def execute(args, log_path: Path, stamp: str) -> int:
    if run_command([sys.executable, str(ROOT / "verify_package.py")]):
        return 1
    if args.check:
        return 0
    base = normalize_ollama_url(args.ollama_url)
    # Shell choice takes precedence over any stale dotenv configuration.
    os.environ["OLLAMA_BASE_URL"] = base
    hardware = preflight(base)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    hardware_path = reports / f"hardware_{stamp}.json"
    hardware_path.write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preflight:
        print("Preflight passed; inference has not been called.")
        return 0
    output = (ROOT / (args.output_dir or ("smoke_runs" if args.smoke else "runs"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".airnrm_mode"
    mode = "smoke" if args.smoke else "full"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() != mode:
        raise ValueError(f"{output} was used for a different mode; choose another --output-dir")
    marker.write_text(mode, encoding="utf-8")
    env = dict(os.environ, LEANOPT_OUT_DIR=str(output), PYTHONUNBUFFERED="1")
    expected = smoke_ids() if args.smoke else list(range(len(query_rows())))
    repeats = 1 if args.smoke else args.repeats
    selected = []
    rc = 0
    attempted = []
    try:
        for repeat in range(repeats):
            attempted.append(repeat)
            print(f"\n=== {'SMOKE' if args.smoke else 'FULL'} repeat {repeat} ===")
            command = [sys.executable, "-u", str(ROOT / "run_notebook.py"), str(NOTEBOOK),
                       "--profile", PROFILE, "--dataset", "Air-NRM-NP-LS",
                       "--data", str(DATA / "query_NP_Flow.csv"), "--repeat", str(repeat)]
            if args.smoke:
                command += ["--rows", ",".join(map(str, expected))]
            elif not args.fresh:
                previous = latest_run(output, repeat)
                if previous:
                    # Explicit ID avoids run_notebook.py's default runs/ lookup.
                    command += ["--resume", previous.name]
            code = run_command(command, env)
            directory = latest_run(output, repeat)
            if directory is not None:
                selected.append(directory)
            passed = validate_run(directory, expected)
            if code or not passed:
                rc = 1
                print("Run failed output checks. See query errors above; outputs remain intact.")
                break
    except BaseException:
        # Save the failure detail in the archived log as well as on the console.
        traceback.print_exc()
        raise
    finally:
        for repeat in attempted:
            directory = latest_run(output, repeat)
            if directory is not None and directory not in selected:
                selected.append(directory)
        archive = ROOT / f"{'smoke' if args.smoke else 'results'}_airnrm_np_{stamp}.tar.gz"
        # Flush the live log before adding it to the archive.
        sys.stdout.flush()
        with tarfile.open(archive, "w:gz") as tar:
            for directory in selected:
                tar.add(directory, arcname=f"runs/{directory.name}")
            tar.add(hardware_path, arcname=f"reports/{hardware_path.name}")
            tar.add(log_path, arcname=f"logs/{log_path.name}")
        print(f"Return bundle: {archive}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="offline checks; no Ollama")
    mode.add_argument("--preflight", action="store_true", help="offline + server/model checks; no inference")
    mode.add_argument("--smoke", action="store_true", help="smallest and largest queries, one repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ollama-url", default=(os.environ.get("OLLAMA_BASE_URL")
                                               or os.environ.get("OLLAMA_URL")
                                               or "http://localhost:11434"))
    parser.add_argument("--output-dir", help="defaults to smoke_runs/ for smoke; runs/ otherwise")
    parser.add_argument("--fresh", action="store_true", help="start fresh instead of resuming full runs")
    parser.add_argument("--dry-run", action="store_true", help="print plan; no imports/server/writes")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.smoke and args.output_dir == "runs":
        parser.error("use a separate smoke output directory, not runs")
    if args.dry_run:
        print(f"Root: {ROOT}\nInterpreter: {sys.executable}\nOllama: {args.ollama_url}")
        print(f"Model: {PROFILE}; dataset: Air-NRM-NP-LS")
        print(f"Mode: {'check' if args.check else 'preflight' if args.preflight else 'smoke' if args.smoke else 'full'}")
        print(f"Queries (1-based): {[i + 1 for i in smoke_ids()] if args.smoke else '1-15'}")
        print(f"Repeats: {1 if args.smoke else args.repeats}")
        return 0
    os.chdir(ROOT)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs / f"airnrm_np_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"Air-NRM-NP-LS / {PROFILE}\nLog: {log_path}")
            try:
                return execute(args, log_path, stamp)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
