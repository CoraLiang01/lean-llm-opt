#!/usr/bin/env python3
"""
dry_run_setup.py
================
Execute a task's setup cells from an empty namespace, with no model involved.

    python dry_run_setup.py large_scale_or
    python dry_run_setup.py loto ["Network Revenue Management"]
    python dry_run_setup.py air_nrm_ls

Embeddings are replaced by a deterministic stand-in, so the FAISS indexes are
really built and every setup cell really runs; what is skipped is the network,
not the code. The chat clients are constructed for real -- constructing one
does not contact the server -- so the agents are assembled exactly as they
would be in a real run. Execution stops at the first cell that calls the batch
function, which is where the drivers stop too.

Why this exists
---------------
`ast.parse` proves a cell is syntactically valid and nothing more. On
2026-08-22 the gpt-oss Large-Scale-OR notebook called `split_by_newline(...)`
while the only definition of it sat inside a commented-out block: valid Python,
fine in the author's session where the function was still in memory from an
earlier execution, and an instant NameError from a clean kernel. All 22 LOTO
seeds died twelve seconds in, on a machine on the other side of the world.

This catches that class of failure in about ten seconds, locally, for free.

Exit code 0 if the setup completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

NOTEBOOKS = {
    "large_scale_or": "LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb",
    "loto": "LEAN_LLM_OPT_gpt_oss_20b_LOTO.ipynb",
    "air_nrm_ls": "LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb",
}

BATCH_CALL_RE = re.compile(
    r"^(?!\s*def\s)\s*(?:\w+\s*(?:,\s*\w+\s*)*=\s*)?"
    r"(Batch_Process_Queries|run_test)\s*\(", re.M)


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "large_scale_or"
    holdout = sys.argv[2] if len(sys.argv) > 2 else None
    if task not in NOTEBOOKS:
        print(f"usage: python dry_run_setup.py {{{'|'.join(NOTEBOOKS)}}} [holdout]",
              file=sys.stderr)
        return 2

    task_dir = ROOT / task
    notebook = NOTEBOOKS[task]
    os.chdir(task_dir)
    sys.path.insert(0, str(task_dir))

    import leanopt_exp as lx
    from langchain_core.embeddings import Embeddings

    class StubEmbeddings(Embeddings):
        """Deterministic 64-d vectors from a hash: same text, same vector."""

        def _vec(self, text: str):
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return [(b - 128) / 128.0 for b in (h * 3)[:64]]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    lx.build_embeddings = lambda cfg: StubEmbeddings()

    print("=" * 66)
    print(f"  dry run: {task} / {notebook}")
    if holdout:
        print(f"  holdout: {holdout}")
    print("  no model, no server, no key -- setup cells only")
    print("=" * 66, flush=True)

    cfg_path = task_dir / "exp_config.yaml"
    original_cfg = None
    if holdout:
        original_cfg = cfg_path.read_text(encoding="utf-8")
        cfg_path.write_text(
            re.sub(r"^(\s*holdout_type:\s*).*$", rf"\g<1>{holdout}",
                   original_cfg, count=1, flags=re.M), encoding="utf-8")

    nb = json.loads((task_dir / notebook).read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    executed, failed = 0, None
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if not src.strip() or src.lstrip().startswith(("!", "%")):
                continue
            if BATCH_CALL_RE.search(src):
                print(f"\nreached the batch cell ({i}) -- setup complete")
                break
            print(f"  cell {i:2d} ...", end="", flush=True)
            try:
                exec(compile(src, f"<cell {i}>", "exec"), ns)   # noqa: S102
            except Exception as e:                              # noqa: BLE001
                print(f" FAILED\n\n  cell {i}: {type(e).__name__}: {e}\n")
                import traceback
                traceback.print_exc()
                failed = i
                break
            executed += 1
            print(" ok")
    finally:
        if original_cfg is not None:
            cfg_path.write_text(original_cfg, encoding="utf-8")

    if failed is not None:
        print(f"\nsetup FAILED at cell {failed} -- do not start a run")
        return 1

    fn = next((f for f in ("Batch_Process_Queries", "run_test")
               if callable(ns.get(f))), None)
    stores = [k for k, v in ns.items() if type(v).__name__ == "FAISS"]
    print(f"\n{executed} setup cell(s) executed with no error")
    print(f"batch entry point : {fn}")
    print(f"FAISS indexes     : {len(stores)}"
          + (f" ({', '.join(sorted(stores))})" if stores else
             "  (this pipeline builds them inside the tools, on demand)"))
    if fn is None:
        print("\nno batch entry point defined -- the drivers cannot call this")
        return 1
    print("\nthe notebook runs from a clean kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
