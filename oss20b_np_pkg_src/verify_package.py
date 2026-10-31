#!/usr/bin/env python3
"""Offline package integrity, data, notebook setup and simulated batch checks."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Test_Dataset/Air_NRM/large_scale"


def verify_manifest() -> None:
    manifest = json.loads((ROOT / "package_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"package file missing: {relative}; extract the whole archive")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"package file changed: {relative}; use a fresh extraction")
    print(f"OK: {len(manifest['sha256'])} packaged files match SHA256", flush=True)


def verify_setup() -> None:
    import pandas as pd
    import leanopt_exp as lx
    from langchain_core.embeddings import Embeddings
    from run_notebook import patch_config_cell, split_at_first_call
    from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
        EXAMPLE_MODEL_CODE, format_parameters, validate_complete_generated_answer,
    )

    class OfflineEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [(byte - 128) / 128 for byte in hashlib.sha256(text.encode()).digest()]

    notebook = json.loads((DATA / "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb").read_text(encoding="utf-8"))
    setup, stop = split_at_first_call(notebook)
    if stop is None:
        raise ValueError("notebook batch cell not found")
    namespace = {"__name__": "__main__"}
    # Every connection is forbidden here. Fake embeddings are only for setup,
    # and all synthetic records live in a deleted temporary directory.
    with tempfile.TemporaryDirectory(prefix="airnrm_np_verify_") as temporary:
        with patch.dict(os.environ, {"LEANOPT_OUT_DIR": temporary}), \
             patch.object(lx, "build_embeddings", return_value=OfflineEmbeddings()), \
             patch("socket.socket.connect", side_effect=RuntimeError("offline check attempted network")):
            for idx, source in setup:
                if source.strip().startswith(("!", "%")):
                    continue
                if "lx.load_config(" in source:
                    source = patch_config_cell(source, "gpt-oss-20b", 0, None, "Air-NRM-NP-LS")
                exec(compile(source, f"<offline cell {idx}>", "exec"), namespace)
            cfg = namespace["CFG"]
            if any(spec.provider != "ollama" or spec.model != "gpt-oss:20b"
                   for spec in cfg.models.values()):
                raise ValueError("an NP role still uses a hosted model")
            if cfg.embedding_provider != "ollama" or cfg.embedding_model != "nomic-embed-text":
                raise ValueError("embeddings do not use local nomic-embed-text")
            # Build every real configured client without issuing any requests.
            for role in cfg.models:
                lx.build_llm(cfg, role)
            # Also construct the real retrieval and ReAct chains: prompt errors
            # often occur here rather than when the notebook defines functions.
            queries = pd.read_csv(DATA / "query_NP_Flow.csv")
            namespace["Classification_Agent"]()
            namespace["FlowAgent"](queries.iloc[0]["Query"])
            is_category = namespace["_is_airnrm_category"]
            for label in ["Sales-Based Linear Programming", "Sales‑Based Linear Programming (SBLP).",
                          "sales based linear programming", "SBLP", "Revenue Management..",
                          "Network Revenue Management"]:
                if not is_category(label):
                    raise ValueError(f"NP routing rejected AirNRM label: {label}")
            for label in ["Transportation Problem", "Assignment Problem", "Others with CSV"]:
                if is_category(label):
                    raise ValueError(f"NP routing accepted unrelated label: {label}")
            namespace["Problemtype"] = lambda query: "Sales‑Based Linear Programming (SBLP)."

            def synthetic_answer(query):
                parameters = namespace["NP_DATA"].assemble(query)
                answer = ("### Objective Function\nMaximize ticket revenue.\n"
                          "### Constraints\nDecision variable domains, capacity, balance, scale, selection, cardinality and flow.\n"
                          "### Generated Code\n```python\n"
                          + format_parameters(parameters) + EXAMPLE_MODEL_CODE + "\n```")
                validate_complete_generated_answer(answer, parameters)
                return answer

            namespace["ProcessPolicyFlow"] = synthetic_answer
            result = namespace["Batch_Process_Queries"](queries)
            namespace["LOG"].close()
            records = namespace["LOG"].records
            if len(result) != 15 or len(records) != 15:
                raise ValueError("offline batch lost a query")
            for record in records:
                if record.status != "ok" or not record.extra.get("code_output"):
                    raise ValueError(f"offline batch failed at {record.instance_id}: {record.errors}")
    print("OK: all setup cells and all 15 simulated NP batches passed; no inference or solving")
    print("Synthetic verification records were deleted; real runs/ was not used.")


def main() -> int:
    os.chdir(ROOT)
    verify_manifest()
    command = [sys.executable, str(DATA / "run_np_lean.py"), "--check"]
    subprocess.run(command, cwd=ROOT, check=True)
    verify_setup()
    print("PASS: package is structurally ready. Server inference is checked by --smoke.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
