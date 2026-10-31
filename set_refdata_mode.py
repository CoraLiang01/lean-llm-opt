#!/usr/bin/env python3
"""
set_refdata_mode.py
===================
Switch the classification step between the two RefData variants and report the
resulting config hash.

    python set_refdata_mode.py --show
    python set_refdata_mode.py full       # every column, as CSVLoader did
    python set_refdata_mode.py trimmed    # description + address + label only

Why this exists
---------------
The original notebooks built the classifier's retrieval corpus with
`CSVLoader(RefData.csv)`, which puts *every* column into every document --
including `Label` (the complete mathematical model) and `Label_Code`. With
k=5 and chain_type="stuff", five full formulations were pasted into the prompt
just to answer "which problem class is this?".

The instrumented version keeps only the columns that bear on the class. Whether
that helps, hurts, or does nothing is an empirical question, and this switches
between the two so it can be answered.

Only `classification.refdata_columns` changes. That single key feeds
`leanopt_exp.load_refdata_docs`, so no notebook and no other setting is
touched -- and because the config file is hashed into every run_id, the two
arms are distinguishable in the logs without anyone having to remember which
was which.

Not identical to the original in one respect: CSVLoader emitted `column: nan`
for empty cells, `load_refdata_docs` skips them. Only 18 of the 96 reference
rows have a Data_address, so the "full" arm here is slightly cleaner than the
code it reproduces. Worth stating if the comparison goes into the paper.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "exp_config.yaml"

MODES = {
    # Everything RefData.csv has -- what CSVLoader loaded.
    "full": ["prompt", "Data_address", "Label", "Label_Code", "Related",
             "New Problem Type"],
    # What the instrumented pipeline uses.
    "trimmed": ["prompt", "Data_address", "New Problem Type"],
}

LINE_RE = re.compile(r"^(\s*refdata_columns:\s*)\[.*?\]\s*$", re.M)


def current_columns() -> list[str]:
    import yaml
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return (raw.get("common", {}).get("classification", {})
               .get("refdata_columns", []))


def describe() -> None:
    cols = current_columns()
    mode = next((m for m, v in MODES.items() if v == cols), "custom")
    print(f"  mode    : {mode}")
    print(f"  columns : {cols}")
    try:
        sys.path.insert(0, str(HERE))
        import leanopt_exp as lx
        for ds in ("Large-Scale-OR",):
            cfg = lx.load_config(CONFIG, model_profile="gpt-4.1",
                                 method="LEAN-LLM-OPT", dataset=ds)
            print(f"  hash    : {cfg.config_hash()}   ({ds})")
        import pandas as pd
        df = lx.load_refdata(cfg)
        docs = lx.load_refdata_docs(cfg)
        chars = sum(len(d.page_content) for d in docs)
        k = cfg.k("refdata")
        print(f"  corpus  : {len(docs)} documents, {chars:,} chars total, "
              f"{chars // max(1, len(docs)):,} chars/doc")
        print(f"  per call: k={k} -> roughly "
              f"{k * chars // max(1, len(docs)):,} chars of retrieved context")
    except Exception as e:                                # noqa: BLE001
        print(f"  (could not load config: {type(e).__name__}: {e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=sorted(MODES),
                    help="which variant to write")
    ap.add_argument("--show", action="store_true",
                    help="report the current setting and exit")
    args = ap.parse_args()

    if args.show or not args.mode:
        describe()
        return 0

    text = CONFIG.read_text(encoding="utf-8")
    if not LINE_RE.search(text):
        raise SystemExit("could not find a refdata_columns line to replace")

    cols = MODES[args.mode]
    rendered = "[" + ", ".join(cols) + "]"
    CONFIG.write_text(LINE_RE.sub(lambda m: m.group(1) + rendered, text,
                                  count=1), encoding="utf-8")
    print(f"set refdata mode -> {args.mode}\n")
    describe()
    print("\n  The config hash above goes into the run_id, so this arm is "
          "identifiable in the logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
