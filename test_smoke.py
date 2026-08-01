"""
Offline smoke test: no API key, no LangChain required.

Fakes the exact objects LangChain hands to the callback handler, then checks
that tokens / calls / cost / latency end up in the log and in the tables.

    python test_smoke.py
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import leanopt_exp as lx

HERE = Path(__file__).parent
RUNS = HERE / "_smoke_runs"
TABLES = HERE / "_smoke_tables"
shutil.rmtree(RUNS, ignore_errors=True)
shutil.rmtree(TABLES, ignore_errors=True)


# --- fake provider responses ------------------------------------------------ #
def openai_style(pt, ct, model, cached=0):
    """Mimics langchain_openai's LLMResult."""
    gen = SimpleNamespace(text="x" * (ct * 4),
                          message=SimpleNamespace(usage_metadata=None,
                                                  response_metadata={}))
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": pt,
                                    "completion_tokens": ct,
                                    "total_tokens": pt + ct,
                                    "prompt_tokens_details": {"cached_tokens": cached}},
                    "model_name": model,
                    "system_fingerprint": "fp_test"},
    )


def ollama_style(pt, ct, model):
    """Mimics ChatOllama: usage only on message.usage_metadata."""
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": pt, "output_tokens": ct,
                        "total_tokens": pt + ct},
        response_metadata={"model": model})
    return SimpleNamespace(generations=[[SimpleNamespace(text="y" * 10, message=msg)]],
                           llm_output={})


def no_usage():
    """Provider gives nothing -> must fall back to an estimate."""
    msg = SimpleNamespace(usage_metadata=None, response_metadata={})
    return SimpleNamespace(
        generations=[[SimpleNamespace(text="z " * 500, message=msg)]],
        llm_output={})


def fake_llm_call(response, model, n_prompt_chars=4000):
    rid = random.random()
    lx.TRACKER.on_chat_model_start(
        {"name": "ChatModel"},
        [[SimpleNamespace(type="human", content="q" * n_prompt_chars)]],
        run_id=rid,
        invocation_params={"model": model, "temperature": 0.0, "top_p": 1.0})
    lx.TRACKER.on_llm_end(response, run_id=rid)


def fake_tool_call(name, ok=True):
    rid = random.random()
    lx.TRACKER.on_tool_start({"name": name}, "some input", run_id=rid)
    if ok:
        lx.TRACKER.on_tool_end("some output", run_id=rid)
    else:
        lx.TRACKER.on_tool_error(ValueError("boom"), run_id=rid)


# --- run 1: full pipeline, GPT-4.1 ------------------------------------------ #
def run_profile(profile, method, model_id, style, dataset="Large-Scale-OR"):
    cfg = lx.load_config(HERE / "exp_config.yaml", model_profile=profile,
                         method=method, dataset=dataset, out_dir=str(RUNS),
                         log_prompts=False, log_raw_responses=False)
    log = lx.RunLogger(cfg)
    types = ["NRM", "RA", "TP", "FLP", "AP"]
    for i in range(6):
        gold = types[i % len(types)]
        with log.instance(instance_id=i, query=f"query {i}", gold_type=gold) as rec:
            with lx.stage("classification"):
                fake_llm_call(style(1200, 40, model_id), model_id)
                fake_tool_call("FileQA")
            pred = gold if i != 3 else "TP"          # one misclassification
            rec.set(pred_type=pred)
            with lx.stage("data_retrieval"):
                for _ in range(3):
                    fake_tool_call("CSVQA")
                    fake_llm_call(style(3000, 300, model_id), model_id)
            with lx.stage("modeling"):
                fake_llm_call(style(6000, 1500, model_id), model_id)
            with lx.stage("codegen"):
                fake_llm_call(style(4000, 1200, model_id), model_id)
            rec.set(model_output="max ...", code_output="import gurobipy",
                    objective_value=123.4)
    return log.close()


agg1 = run_profile("gpt-4.1", "LEAN-LLM-OPT", "gpt-4.1-2025-04-14", openai_style)
agg2 = run_profile("gpt-oss-20b", "LEAN-LLM-OPT", "gpt-oss:20b", ollama_style)

# --- run 2: single-call baseline (unknown price + no usage metadata) --------- #
cfg = lx.load_config(HERE / "exp_config.yaml", model_profile="gpt-4.1",
                     method="Base-SingleCall", dataset="Large-Scale-OR",
                     out_dir=str(RUNS), log_prompts=False)
log = lx.RunLogger(cfg)
for i in range(6):
    with log.instance(instance_id=i, query=f"query {i}", gold_type="NRM") as rec:
        with lx.stage("single_call"):
            fake_llm_call(no_usage(), "gpt-4.1-2025-04-14")
        rec.set(pred_type=None, code_output="import gurobipy")
agg3 = log.close()

# --- assertions ------------------------------------------------------------- #
first = json.loads(
    (RUNS / agg1["run_id"] / "instances.jsonl").read_text().splitlines()[0])
s = first["summary"]
assert s["n_llm_calls"] == 6, s
assert s["n_tool_calls"] == 4, s
assert s["prompt_tokens"] == 1200 + 3 * 3000 + 6000 + 4000, s
assert s["cost_usd"] > 0, s
assert set(s["by_stage"]) == {"classification", "data_retrieval", "modeling",
                              "codegen"}, s
assert agg2["total_cost_usd"] == 0.0, "local model must cost $0"
assert agg1["total_n_llm_calls"] == 36

est = json.loads(
    (RUNS / cfg.run_id / "instances.jsonl").read_text().splitlines()[0])
assert est["summary"]["n_estimated_token_calls"] == 1, est["summary"]
assert est["summary"]["prompt_tokens"] > 0


# --- embedding wrapper: build an index AND query it ------------------------- #
# Regression test for "'CountingEmbeddings' object is not callable": FAISS only
# checks isinstance(embedding, Embeddings) at QUERY time, so building an index
# succeeds even when the wrapper is wrong and the first retrieval blows up.
print("\n--- embedding wrapper round-trip ---")
try:
    from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
    from langchain_community.vectorstores import FAISS

    cfg_e = lx.load_config(HERE / "exp_config.yaml", model_profile="gpt-4.1",
                           method="EMB", dataset="Large-Scale-OR",
                           out_dir=str(RUNS), log_prompts=False)
    emb = lx.CountingEmbeddings(DeterministicFakeEmbedding(size=32), cfg_e,
                                cfg_e.embedding_price_per_1m)
    assert isinstance(emb, Embeddings), \
        "CountingEmbeddings must subclass langchain_core Embeddings"

    log_e = lx.RunLogger(cfg_e)
    with log_e.instance(instance_id=0, query="q") as rec:
        with lx.stage("classification"):
            store = FAISS.from_documents(lx.load_refdata_docs(cfg_e), emb)
            hits = store.as_retriever(search_kwargs={"k": 3}).invoke("assign engineers")
            assert len(hits) == 3, hits
            emb.embed_query("hello")
    log_e.close()
    s_e = log_e.records[-1].summary()
    assert s_e["n_embedding_calls"] >= 2, s_e
    assert s_e["embedding_tokens"] > 0, s_e
    print(f"  index build + query + embed_query OK "
          f"({s_e['n_embedding_calls']} embedding calls, "
          f"{s_e['embedding_tokens']:,} tokens)")
except ImportError as e:
    print(f"  skipped (langchain/faiss not installed here): {e}")

print("\n--- assertions passed ---")

# --- gold labels + aggregation ---------------------------------------------- #
import pandas as pd
gold = pd.DataFrame({
    "dataset": ["Large-Scale-OR"] * 6,
    "instance_id": list(range(6)),
    "correct_optimal": [1, 1, 1, 0, 1, 0],
    "correct_formulation": [1, 1, 0, 0, 1, 0],
})
gold.to_csv(HERE / "_smoke_gold.csv", index=False)

r = subprocess.run([sys.executable, str(HERE / "aggregate_runs.py"), str(RUNS),
                    "-o", str(TABLES), "--gold", str(HERE / "_smoke_gold.csv")],
                   capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-3000:])
assert r.returncode == 0
print("\nfiles:", sorted(p.name for p in TABLES.iterdir()))
print(pd.read_csv(TABLES / "classification.csv").to_string(index=False))
print(pd.read_csv(TABLES / "stage_breakdown.csv").to_string(index=False))
