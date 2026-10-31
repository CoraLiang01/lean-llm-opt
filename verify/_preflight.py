#!/usr/bin/env python3
"""
_preflight.py
=============
Refuse to spend money on a run whose config is not the one being tested.

Called first by every verify script. It asserts the settings that must not
drift, prints the ones that are being varied on purpose, and says where the API
key will come from. Getting the first kind wrong is expensive and silent: the
run completes, the numbers look plausible, and nothing in the output says the
pipeline under test was not the intended one.

Asserting the second kind is its own failure mode -- it refuses to run half of
a planned A/B -- so the classifier model is reported rather than pinned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# Asserted, because drift here is a silent wrong-pipeline bug rather than an
# experiment: text-embedding-3-small was what broke classifier routing (NRM
# reached its own agent 3 times out of 25, against 22-24 on ada-002).
EXPECTED = {
    "embedding_model": "text-embedding-ada-002",
}

# The classifier model is deliberately NOT asserted. It is the A/B variable of
# 2026-08-23 -- gpt-4.1 in Run A, gpt-4 in Run B -- so pinning it here refuses
# to run whichever half of the comparison it is not set to, which is what
# happened on the first attempt at Run B. It is printed below instead, so the
# run's own output records which arm this was.


def check_environment() -> list[str]:
    """The interpreter must be the one the pipeline's dependencies live in.

    `python3` on this machine resolves to anaconda base, where langchain-core
    is 0.1.53 and `from langchain_openai import ChatOpenAI` raises ImportError
    on LangSmithParams. The pipeline needs the `leanopt` env (langchain 1.2.3 /
    langchain-classic 1.0.1, as pinned in requirements.txt). Import the real
    thing here rather than comparing version strings: a run that dies fifteen
    setup cells in has already cost the classifier calls it made on the way.
    """
    problems = []
    for mod, why in (("langchain_openai", "ChatOpenAI / OpenAIEmbeddings"),
                     ("langchain_classic", "RetrievalQA, agents"),
                     ("faiss", "vector stores"),
                     ("gurobipy", "solving the generated models"),
                     ("openpyxl", "the March baseline spreadsheet")):
        try:
            __import__(mod)
        except Exception as e:                                # noqa: BLE001
            problems.append(f"cannot import {mod} ({why}): "
                            f"{type(e).__name__}: {e}")
    if problems:
        problems.append("this is almost always the wrong interpreter -- "
                        "run `conda activate leanopt` first "
                        f"(currently {sys.executable})")
    return problems


def main() -> int:
    env_problems = check_environment()
    if env_problems:
        print("refusing to run:")
        for p in env_problems:
            print(f"  ! {p}")
        return 1

    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env", override=True)
    except ImportError:
        pass

    import leanopt_exp as lx
    cfg = lx.load_config(str(HERE / "exp_config.yaml"),
                         model_profile="gpt-4.1",
                         method="LEAN-LLM-OPT",
                         dataset="Large-Scale-OR",
                         repeat_index=0)

    actual = {
        "embedding_model": cfg.embedding_model,
        "classifier_model": cfg.spec("classifier").model,
    }
    n_refdata_cols = len(lx.load_refdata_docs(cfg)[0].page_content.splitlines())

    print("config under test")
    print("-" * 60)
    bad = []
    for k, want in EXPECTED.items():
        got = actual[k]
        flag = "ok" if got == want else "MISMATCH"
        if got != want:
            bad.append(f"{k}: expected {want!r}, got {got!r}")
        print(f"  {k:<18} {got:<28} {flag}")
    for role in ("classifier", "modeler", "data_agent", "coder"):
        mt = cfg.spec(role).max_tokens
        print(f"  {role:<18} {cfg.spec(role).model:<28} "
              f"max_tokens={'unset' if mt is None else mt}")
    print(f"  refdata doc fields {n_refdata_cols}"
          f"  (2 would mean the trimmed-column fallback is back)")

    # Not a failure. As of 2026-08-23 the key is deliberately not stored in
    # .env; lx.ensure_api_keys() prompts for it with hidden input when the
    # config cell runs, and never writes it anywhere. Refusing here would block
    # the run before it ever got the chance to ask.
    key = os.environ.get("OPENAI_API_KEY", "")
    if len(key.strip()) >= 20 and not key.strip().endswith("..."):
        print("\n  OPENAI_API_KEY      found in the environment")
    else:
        print("\n  OPENAI_API_KEY      not set -- you will be prompted "
              "(hidden input) once the run starts")

    if bad:
        print("\nrefusing to run:")
        for b in bad:
            print(f"  ! {b}")
        return 1
    print("\npreflight ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
