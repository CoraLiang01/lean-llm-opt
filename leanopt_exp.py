"""
leanopt_exp.py
==============
Unified experiment configuration + run logging + resource accounting for
LEAN-LLM-OPT.

Addresses reviewer requests:
  * Referee 1, Q4  -- "report exact prompts, model versions, retrieval settings,
                      decoding parameters, failure handling"
  * Referee 2, Q3  -- "prompt/completion tokens, tool calls, API cost, latency
                      ... needed for fair comparison with single-call baselines"

Design goals
------------
1. ONE config object drives every LLM / embedding / retriever in every notebook.
   No more `model_name="gpt-4"` in the classifier and `"gpt-4.1"` in the modeler.
2. Every LLM call and every tool call is counted automatically by a LangChain
   callback handler that is attached at construction time, so notebook code
   does not need per-call instrumentation.
3. Every benchmark instance produces one JSON line with the full resource
   footprint + provenance, so results are reproducible and aggregatable.

Usage (minimal)
---------------
    import leanopt_exp as lx

    CFG = lx.load_config("exp_config.yaml", model_profile="gpt-4.1",
                         method="LEAN-LLM-OPT", dataset="Large-Scale-OR")
    LOG = lx.RunLogger(CFG)

    llm_cls  = lx.build_llm(CFG, "classifier")   # replaces ChatOpenAI(...)
    llm_mod  = lx.build_llm(CFG, "modeler")
    llm_code = lx.build_llm(CFG, "coder")
    emb      = lx.build_embeddings(CFG)

    for i, row in test.iterrows():
        with LOG.instance(instance_id=i, query=row["Query"],
                          gold_type=row.get("Type")) as rec:
            with lx.stage("classification"):
                ptype = classify(row["Query"])
            rec.set(pred_type=ptype)
            with lx.stage("modeling"):
                model_text = get_NRM_response(...)
            with lx.stage("codegen"):
                code = get_code(model_text, ptype)
            rec.set(model_output=model_text, code_output=code)

    LOG.close()

Then:  python aggregate_runs.py runs/ -o tables/
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ExpConfig", "ModelSpec", "load_config", "build_llm", "build_embeddings",
    "build_retriever", "agent_kwargs", "UsageTracker", "TRACKER", "stage",
    "ensure_api_keys", "load_refdata", "load_refdata_docs",
    "refdata_token_report",
    "RunLogger", "InstanceRecord", "call_with_retry", "environment_manifest",
    "dataset_fingerprint",
]

# Repository root. Dataset paths in exp_config.yaml are written relative to it,
# and resolving them here rather than against the caller's cwd keeps them valid
# whether the entry point is a notebook, a script, or run_notebook.py.
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 0.  Configuration objects
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    """Decoding + pricing spec for one LLM role."""
    provider: str                      # "openai" | "ollama" | "google" | "anthropic"
    model: str                         # exact model id, e.g. "gpt-4.1-2025-04-14"
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    seed: Optional[int] = 20250101     # OpenAI honours `seed`; ignored elsewhere
    max_tokens: Optional[int] = None
    # USD per 1M tokens. Local models -> 0.0 (we log GPU wall-clock instead).
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0
    price_cached_in_per_1m: float = 0.0
    base_url: Optional[str] = None     # for ollama / vLLM / OpenRouter / azure
    # Which environment variable holds the key. Lets an OpenAI-compatible
    # gateway (OpenRouter, vLLM, Azure) use its own key without touching
    # OPENAI_API_KEY.
    api_key_env: str = "OPENAI_API_KEY"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpConfig:
    """Everything an experiment run needs. Serialised verbatim into the log."""
    # --- identity -------------------------------------------------------- #
    run_id: str = ""
    method: str = "LEAN-LLM-OPT"       # or "Ablation-NoWorkflow", "GPT-5.2-single", ...
    dataset: str = ""
    model_profile: str = ""            # key into exp_config.yaml:model_profiles
    notes: str = ""
    # exp_config.yaml:dataset_columns -- see resolve_columns()
    dataset_columns: Dict[str, Any] = field(default_factory=dict)
    dataset_path: Optional[str] = None

    # --- models ---------------------------------------------------------- #
    # roles: classifier | modeler | data_agent | coder
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: Optional[str] = None
    embedding_api_key_env: str = "OPENAI_API_KEY"
    # USD per 1M tokens for the embedding model (0 for local models).
    embedding_price_per_1m: float = 0.02

    # --- retrieval ------------------------------------------------------- #
    # Per-call-site retrieval settings, e.g.
    #   retrievers: {refdata: {k: 5}, data_ap: {k: 1, max_tokens_limit: 400}}
    # Values mirror what the notebooks actually use; nothing is silently
    # normalised, because k is load-bearing here (k=1000 pulls whole tables).
    retrievers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retriever_k: Dict[str, int] = field(default_factory=lambda: {"default": 5})
    retriever_max_tokens_limit: Optional[int] = None
    retriever_search_type: str = "similarity"

    # --- classification ---------------------------------------------------- #
    # Which RefData columns the classifier is allowed to see. Dropping `Label`
    # / `Label_Code` removes the full formulations from the prompt: they cost a
    # lot of tokens and say nothing about the problem class.
    classification: Dict[str, Any] = field(default_factory=lambda: {
        "refdata_file": "Large_Scale_Or_Files/RefData.csv",
        "refdata_columns": ["prompt", "New Problem Type"],
        "label_column": "New Problem Type",
    })

    # --- agent ----------------------------------------------------------- #
    agent_type: str = "ZERO_SHOT_REACT_DESCRIPTION"
    agent_max_iterations: int = 8
    agent_max_execution_time: Optional[float] = 300.0
    agent_early_stopping_method: str = "force"
    handle_parsing_errors: bool = True
    verbose: bool = False

    # --- execution ------------------------------------------------------- #
    seed: int = 20250101
    n_repeats: int = 1                 # repeat each instance -> variance bars
    repeat_index: int = 0
    max_retries: int = 3
    retry_backoff_s: float = 5.0
    sleep_between_instances_s: float = 0.0   # was hard-coded time.sleep(15)
    solver_time_limit_s: float = 300.0
    # Hard stop for a single run. 0 disables it. The check happens after each
    # instance, so the overshoot is bounded by one instance.
    budget_usd_per_run: float = 0.0

    # --- logging --------------------------------------------------------- #
    out_dir: str = "runs"
    log_prompts: bool = True           # dumps full prompts -> referee 1 Q4
    log_raw_responses: bool = True
    token_fallback_encoder: str = "o200k_base"

    # --- provenance (filled automatically) ------------------------------- #
    manifest: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("manifest", None)
        d.pop("run_id", None)
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def spec(self, role: str) -> ModelSpec:
        if role not in self.models:
            raise KeyError(
                f"role '{role}' not in config.models (have: {list(self.models)})"
            )
        return self.models[role]

    def k(self, name: str = "default") -> int:
        if name in self.retrievers and "k" in self.retrievers[name]:
            return self.retrievers[name]["k"]
        return self.retriever_k.get(name, self.retriever_k.get("default", 5))

    def retriever_cfg(self, name: str) -> Dict[str, Any]:
        cfg = dict(self.retrievers.get(name, {}))
        cfg.setdefault("k", self.k(name))
        if "max_tokens_limit" not in cfg and self.retriever_max_tokens_limit:
            cfg["max_tokens_limit"] = self.retriever_max_tokens_limit
        cfg.setdefault("search_type", self.retriever_search_type)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _pkg_versions() -> Dict[str, str]:
    names = ["langchain", "langchain_core", "langchain_openai", "langchain_ollama",
             "langchain_community", "langchain_classic", "openai", "faiss",
             "gurobipy", "pandas", "numpy", "tiktoken"]
    out = {}
    for n in names:
        try:
            import importlib.metadata as md
            out[n] = md.version(n.replace("_", "-"))
        except Exception:
            try:
                mod = __import__(n)
                out[n] = getattr(mod, "__version__", "unknown")
            except Exception:
                pass
    return out


def dataset_fingerprint(path: str | Path | None) -> Dict[str, Any]:
    """Identify the dataset file a run actually read.

    `config_sha256` fixes the settings but says nothing about the data: the
    config records a *path*, so editing the file it points at leaves the hash
    untouched. Two runs can therefore carry an identical config hash and have
    been measured on different problems -- which happened here on 2026-08-02,
    when a 101-instance file was temporarily replaced by a 14-instance one and
    nothing in the log showed it.

    So hash the bytes too. Any edit to the dataset changes `sha256`, and
    `rows` makes the common case (a different number of instances) readable
    without comparing hashes by eye.

    A missing file is recorded rather than raised on: this runs at config load,
    long before anything would actually read the data, and failing here would
    turn a dry-run or a preflight import into an error.
    """
    if not path:
        return {"path": None, "present": False}
    p = Path(path)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists():
        return {"path": str(path), "present": False}

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    out: Dict[str, Any] = {
        "path": str(path),
        "present": True,
        "sha256": h.hexdigest(),
        "bytes": p.stat().st_size,
        "mtime_utc": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(
            timespec="seconds") + "Z",
    }
    # Row count via pandas rather than counting newlines: several of these CSVs
    # embed newlines inside quoted fields (the Air-NRM queries do), so a line
    # count would be wrong in exactly the files it matters for.
    try:
        import pandas as pd
        out["rows"] = int(len(pd.read_csv(p)))
    except Exception:                                     # noqa: BLE001
        pass
    return out


def environment_manifest() -> Dict[str, Any]:
    """Provenance block -> answers 'report model versions / settings'."""
    try:
        import gurobipy as gp
        gurobi_v = ".".join(str(x) for x in gp.gurobi.version())
    except Exception:
        gurobi_v = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": _pkg_versions(),
        "gurobi": gurobi_v,
    }


CANONICAL_COLUMNS = ("query", "gold_objective", "gold_type",
                     "dataset_address", "size_class")


def resolve_columns(cfg: "ExpConfig", columns) -> Dict[str, Optional[str]]:
    """Map canonical field names onto this dataset's actual column names.

    The benchmark CSVs were assembled from different sources and disagree on
    what to call the same thing: Large-Scale-OR stores the ground-truth optimum
    in `Label-objective`, every small-scale set stores it in `Label`. Reading
    one fixed name means the other datasets silently arrive without ground
    truth, and an instance with no ground truth scores the same as a wrong one
    -- a column-name mismatch dressed up as a modelling failure.

    Resolution order, per field:
      1. `dataset_columns.overrides.<dataset>` if it names this field
         (an explicit `null` there means "this dataset genuinely has none")
      2. the first `dataset_columns.candidates.<field>` entry present in the CSV
      3. None

    Returns every canonical name, with None where nothing matched, so callers
    can tell "absent" from "not looked for".
    """
    cols = list(columns)
    spec = cfg.dataset_columns or {}
    candidates = spec.get("candidates", {}) or {}
    override = (spec.get("overrides", {}) or {}).get(cfg.dataset, {}) or {}

    out: Dict[str, Optional[str]] = {}
    for field_name in CANONICAL_COLUMNS:
        if field_name in override:
            name = override[field_name]
            out[field_name] = name if (name and name in cols) else None
            continue
        out[field_name] = next(
            (c for c in candidates.get(field_name, []) if c in cols), None)
    return out


def describe_columns(mapping: Dict[str, Optional[str]]) -> str:
    """One-line, log-friendly rendering of resolve_columns() output."""
    return "  ".join(f"{k}={v or '-'}" for k, v in mapping.items())


def load_config(path: str | Path,
                model_profile: str,
                method: str = "LEAN-LLM-OPT",
                dataset: str = "",
                repeat_index: int = 0,
                **overrides) -> ExpConfig:
    """Load exp_config.yaml (or .json) and materialise one ExpConfig."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(raw_text)
    else:
        raw = json.loads(raw_text)

    common = dict(raw.get("common", {}))
    profiles = raw.get("model_profiles", {})
    if model_profile not in profiles:
        raise KeyError(f"model_profile '{model_profile}' not in {list(profiles)}")
    prof = profiles[model_profile]

    # role -> ModelSpec, inheriting profile defaults
    defaults = dict(prof.get("defaults", {}))
    models: Dict[str, ModelSpec] = {}
    for role, spec in prof.get("roles", {}).items():
        merged = {**defaults, **(spec or {})}
        merged = {k: v for k, v in merged.items()
                  if k in {f.name for f in dataclasses.fields(ModelSpec)}}
        models[role] = ModelSpec(**merged)

    cfg_kwargs = {k: v for k, v in common.items()
                  if k in {f.name for f in dataclasses.fields(ExpConfig)}}
    cfg_kwargs.update({
        "models": models,
        "model_profile": model_profile,
        "method": method,
        "dataset": dataset,
        "repeat_index": repeat_index,
    })
    for key in ("embedding_provider", "embedding_model", "embedding_base_url",
                "embedding_api_key_env", "embedding_price_per_1m"):
        if key in prof:
            cfg_kwargs[key] = prof[key]
    cfg_kwargs.update(overrides)

    cfg = ExpConfig(**cfg_kwargs)
    # Carried on the config so resolve_columns() needs no second file read and
    # so the mapping in force is serialised into the run log with everything
    # else -- which column supplied the ground truth is part of how a number
    # was produced.
    cfg.dataset_columns = raw.get("dataset_columns", {}) or {}
    cfg.dataset_path = (raw.get("datasets", {}) or {}).get(dataset)
    cfg.manifest = environment_manifest()
    cfg.manifest["config_file"] = str(path.resolve())
    cfg.manifest["config_sha256"] = hashlib.sha256(raw_text.encode()).hexdigest()
    # config_sha256 covers the settings; this covers the data they point at.
    cfg.manifest["dataset"] = dataset_fingerprint(cfg.dataset_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.run_id = (f"{method}__{model_profile}__{dataset or 'na'}"
                  f"__r{repeat_index}__{stamp}__{cfg.config_hash()}")
    return cfg


# --------------------------------------------------------------------------- #
# 1.  Usage tracking callback
# --------------------------------------------------------------------------- #

_CURRENT_RECORD: contextvars.ContextVar[Optional["InstanceRecord"]] = \
    contextvars.ContextVar("leanopt_current_record", default=None)
_CURRENT_STAGE: contextvars.ContextVar[str] = \
    contextvars.ContextVar("leanopt_current_stage", default="unassigned")


@contextlib.contextmanager
def stage(name: str):
    """Attribute every LLM/tool call inside the block to a pipeline stage."""
    token = _CURRENT_STAGE.set(name)
    try:
        yield
    finally:
        _CURRENT_STAGE.reset(token)


def _extract_usage(response) -> Dict[str, Any]:
    """
    Normalise token usage across providers.

    OpenAI (langchain_openai)  -> response.llm_output["token_usage"]
    Ollama / others            -> generations[0][0].message.usage_metadata
    Nothing available          -> caller falls back to tiktoken estimate.
    """
    usage = {"prompt_tokens": None, "completion_tokens": None,
             "total_tokens": None, "cached_prompt_tokens": 0,
             "reasoning_tokens": 0, "token_source": "none",
             "system_fingerprint": None, "model_name": None}

    llm_out = getattr(response, "llm_output", None) or {}
    tu = llm_out.get("token_usage") or llm_out.get("usage") or {}
    if tu:
        usage["prompt_tokens"] = tu.get("prompt_tokens") or tu.get("input_tokens")
        usage["completion_tokens"] = (tu.get("completion_tokens")
                                      or tu.get("output_tokens"))
        usage["total_tokens"] = tu.get("total_tokens")
        det = tu.get("prompt_tokens_details") or {}
        if isinstance(det, dict):
            usage["cached_prompt_tokens"] = det.get("cached_tokens", 0) or 0
        cdet = tu.get("completion_tokens_details") or {}
        if isinstance(cdet, dict):
            usage["reasoning_tokens"] = cdet.get("reasoning_tokens", 0) or 0
        usage["token_source"] = "provider"
    usage["system_fingerprint"] = llm_out.get("system_fingerprint")
    usage["model_name"] = llm_out.get("model_name") or llm_out.get("model")

    if usage["prompt_tokens"] is None:
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) or {}
            if um:
                usage["prompt_tokens"] = um.get("input_tokens")
                usage["completion_tokens"] = um.get("output_tokens")
                usage["total_tokens"] = um.get("total_tokens")
                idet = um.get("input_token_details") or {}
                usage["cached_prompt_tokens"] = idet.get("cache_read", 0) or 0
                odet = um.get("output_token_details") or {}
                usage["reasoning_tokens"] = odet.get("reasoning", 0) or 0
                usage["token_source"] = "usage_metadata"
                meta = getattr(msg, "response_metadata", None) or {}
                usage["model_name"] = usage["model_name"] or meta.get("model_name") \
                    or meta.get("model")
        except Exception:
            pass

    if usage["total_tokens"] is None and usage["prompt_tokens"] is not None:
        usage["total_tokens"] = (usage["prompt_tokens"] or 0) + \
                                (usage["completion_tokens"] or 0)
    return usage


_ENCODER_CACHE: Dict[str, Any] = {}


def _estimate_tokens(text: str, encoding: str = "o200k_base") -> int:
    if not text:
        return 0
    try:
        import tiktoken
        if encoding not in _ENCODER_CACHE:
            _ENCODER_CACHE[encoding] = tiktoken.get_encoding(encoding)
        return len(_ENCODER_CACHE[encoding].encode(text))
    except Exception:
        return max(1, len(text) // 4)


class UsageTracker:
    """
    LangChain callback handler (duck-typed; inherits BaseCallbackHandler when
    available) that records every LLM and tool call into the *current*
    InstanceRecord. One global singleton is attached to all LLMs at build time,
    so notebooks need no per-call instrumentation.
    """

    raise_error = False
    run_inline = True
    ignore_llm = False
    ignore_chain = False
    ignore_agent = False
    ignore_retriever = False
    ignore_chat_model = False
    ignore_retry = False
    ignore_custom_event = True

    def __init__(self, cfg: Optional[ExpConfig] = None):
        self.cfg = cfg
        self._starts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def bind(self, cfg: ExpConfig):
        self.cfg = cfg
        return self

    # -- price lookup ---------------------------------------------------- #
    def _price(self, model_name: Optional[str]) -> Optional[ModelSpec]:
        if not self.cfg:
            return None
        if model_name:
            for spec in self.cfg.models.values():
                if spec.model == model_name or model_name.startswith(spec.model):
                    return spec
        # fall back to the modeler spec (dominant cost) if unmatched
        return self.cfg.models.get("modeler") or next(
            iter(self.cfg.models.values()), None)

    # -- LLM ------------------------------------------------------------- #
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._begin(run_id, kind="llm", prompts=list(prompts or []),
                    serialized=serialized, kwargs=kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        flat = []
        for conv in messages or []:
            flat.append("\n".join(
                f"{getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
                for m in conv))
        self._begin(run_id, kind="chat", prompts=flat,
                    serialized=serialized, kwargs=kwargs)

    def _begin(self, run_id, kind, prompts, serialized, kwargs):
        key = str(run_id)
        invocation = (kwargs or {}).get("invocation_params", {}) or {}
        with self._lock:
            if key in self._starts:
                return          # handler seen twice (global + local): ignore
            self._starts[key] = {
                "t0": time.perf_counter(),
                "kind": kind,
                "prompts": prompts,
                "stage": _CURRENT_STAGE.get(),
                "requested_model": invocation.get("model") or
                                   invocation.get("model_name"),
                "temperature": invocation.get("temperature"),
                "top_p": invocation.get("top_p"),
            }

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        key = str(run_id)
        with self._lock:
            st = self._starts.pop(key, None)
        rec = _CURRENT_RECORD.get()
        if rec is None or st is None:
            # st is None => this run_id was already accounted for (the handler
            # is registered both globally and on the model object)
            return
        latency = time.perf_counter() - st["t0"]
        usage = _extract_usage(response)

        completion_text = ""
        try:
            completion_text = "".join(
                getattr(g, "text", "") or ""
                for gen in response.generations for g in gen)
        except Exception:
            pass

        enc = self.cfg.token_fallback_encoder if self.cfg else "o200k_base"
        if usage["prompt_tokens"] is None:
            usage["prompt_tokens"] = sum(
                _estimate_tokens(p, enc) for p in st.get("prompts", []))
            usage["completion_tokens"] = _estimate_tokens(completion_text, enc)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["token_source"] = "estimated"

        model_name = usage["model_name"] or st.get("requested_model")
        spec = self._price(model_name)
        p_in = (usage["prompt_tokens"] or 0) - (usage["cached_prompt_tokens"] or 0)
        cost, price_missing = 0.0, False
        if spec is None:
            price_missing = True
        elif spec.price_in_per_1m is None or spec.price_out_per_1m is None:
            # price not filled in yet -> record 0 but flag it, never fake a number
            price_missing = True
        else:
            cost = (max(p_in, 0) / 1e6) * spec.price_in_per_1m \
                 + ((usage["cached_prompt_tokens"] or 0) / 1e6) * (spec.price_cached_in_per_1m or 0.0) \
                 + ((usage["completion_tokens"] or 0) / 1e6) * spec.price_out_per_1m

        call = {
            "type": "llm",
            "stage": st.get("stage", "unassigned"),
            "model": model_name,
            "requested_model": st.get("requested_model"),
            "temperature": st.get("temperature"),
            "top_p": st.get("top_p"),
            "system_fingerprint": usage["system_fingerprint"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_prompt_tokens": usage["cached_prompt_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "token_source": usage["token_source"],
            "latency_s": latency,
            "cost_usd": cost,
            "price_missing": price_missing,
            "ok": True,
        }
        if self.cfg and self.cfg.log_prompts:
            call["prompts"] = st.get("prompts", [])
        if self.cfg and self.cfg.log_raw_responses:
            call["completion"] = completion_text
        rec.add_call(call)

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        key = str(run_id)
        with self._lock:
            st = self._starts.pop(key, None)
        rec = _CURRENT_RECORD.get()
        if rec is None:
            return
        rec.add_call({
            "type": "llm",
            "stage": st.get("stage", "unassigned"),
            "model": st.get("requested_model"),
            "latency_s": (time.perf_counter() - st["t0"]) if st else None,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cost_usd": 0.0,
        })

    # -- tools ----------------------------------------------------------- #
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        with self._lock:
            self._starts[str(run_id)] = {
                "t0": time.perf_counter(),
                "kind": "tool",
                "name": (serialized or {}).get("name", "unknown_tool"),
                "stage": _CURRENT_STAGE.get(),
                "input": str(input_str)[:2000],
            }

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        with self._lock:
            st = self._starts.pop(str(run_id), None)
        rec = _CURRENT_RECORD.get()
        if rec is None or st is None:
            return
        rec.add_call({
            "type": "tool",
            "stage": st["stage"],
            "tool_name": st["name"],
            "latency_s": time.perf_counter() - st["t0"],
            "ok": True,
            "input": st["input"],
            "output_chars": len(str(output)),
        })

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        with self._lock:
            st = self._starts.pop(str(run_id), None)
        rec = _CURRENT_RECORD.get()
        if rec is None or st is None:
            return
        rec.add_call({
            "type": "tool", "stage": st["stage"], "tool_name": st["name"],
            "latency_s": time.perf_counter() - st["t0"], "ok": False,
            "error": f"{type(error).__name__}: {error}",
        })

    # -- retriever ------------------------------------------------------- #
    def on_retriever_start(self, serialized, query, *, run_id=None, **kwargs):
        with self._lock:
            self._starts[str(run_id)] = {
                "t0": time.perf_counter(), "kind": "retriever",
                "stage": _CURRENT_STAGE.get(), "query": str(query)[:1000],
            }

    def on_retriever_end(self, documents, *, run_id=None, **kwargs):
        with self._lock:
            st = self._starts.pop(str(run_id), None)
        rec = _CURRENT_RECORD.get()
        if rec is None or st is None:
            return
        rec.add_call({
            "type": "retriever", "stage": st["stage"],
            "latency_s": time.perf_counter() - st["t0"],
            "n_docs": len(documents or []), "ok": True,
            "query": st["query"],
        })

    def on_retriever_error(self, error, *, run_id=None, **kwargs):
        with self._lock:
            self._starts.pop(str(run_id), None)

    # -- unused hooks (kept so LangChain never crashes) ------------------- #
    def on_llm_new_token(self, *a, **k):
        pass

    def on_chain_start(self, *a, **k):
        pass

    def on_chain_end(self, *a, **k):
        pass

    def on_chain_error(self, *a, **k):
        pass

    def on_agent_action(self, action, *, run_id=None, **kwargs):
        rec = _CURRENT_RECORD.get()
        if rec is not None:
            rec.add_trace({"agent_action": getattr(action, "tool", None),
                           "stage": _CURRENT_STAGE.get()})

    def on_agent_finish(self, *a, **k):
        pass

    def on_text(self, *a, **k):
        pass

    def on_retry(self, retry_state, *, run_id=None, **kwargs):
        rec = _CURRENT_RECORD.get()
        if rec is not None:
            rec.n_provider_retries += 1

    def on_custom_event(self, *a, **k):
        pass


# make it a real BaseCallbackHandler when langchain is importable
try:  # pragma: no cover
    from langchain_core.callbacks.base import BaseCallbackHandler as _BCH

    class UsageTracker(UsageTracker, _BCH):  # type: ignore[no-redef]
        pass
except Exception:  # pragma: no cover
    pass


TRACKER = UsageTracker()


# --------------------------------------------------------------------------- #
# Make the tracker INHERITABLE.
#
# Handlers passed as `callbacks=[...]` to a chain or agent are *local*: they see
# that run but are not propagated to child runs. Tools are child runs of the
# AgentExecutor, so tool calls were being missed entirely (LLM calls still
# showed up because the handler is attached to the LLM object itself).
#
# register_configure_hook adds the handler to every callback manager LangChain
# configures, at every level: chains, agents, tools, retrievers, LLMs.
# --------------------------------------------------------------------------- #
_TRACKER_VAR: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "leanopt_tracker", default=None)

try:  # pragma: no cover
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(_TRACKER_VAR, True)
    _TRACKER_VAR.set(TRACKER)
    _GLOBAL_HOOK = True
except Exception:  # pragma: no cover
    _GLOBAL_HOOK = False


# --------------------------------------------------------------------------- #
# 2.  Model / retriever factories
# --------------------------------------------------------------------------- #

def load_refdata(cfg: ExpConfig) -> "Any":
    """
    Read the reference table, tolerating the fact that RefData.csv has, at
    times, actually held xlsx bytes. Returns a pandas DataFrame.
    """
    import pandas as pd
    path = Path(cfg.classification.get(
        "refdata_file", "Large_Scale_Or_Files/RefData.csv"))
    if path.read_bytes()[:2] == b"PK":          # xlsx wearing a .csv name
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_refdata_docs(cfg: ExpConfig, columns: Optional[List[str]] = None):
    """
    Build the documents the CLASSIFICATION step retrieves over.

    The default CSVLoader put every column into each document, including
    `Label` (the full mathematical model) and `Label_Code`. With k=5 and
    chain_type="stuff" that means five complete formulations are pasted into
    the prompt just to answer "which problem class is this?" -- a large amount
    of context that carries no information about the class.

    This keeps only the columns listed in
    exp_config.yaml:classification.refdata_columns (by default the problem
    description and the class label), which is what the task actually needs.

    Returns: List[Document]
    """
    from langchain_core.documents import Document

    df = load_refdata(cfg)
    cols = columns or cfg.classification.get(
        "refdata_columns", ["prompt", "New Problem Type"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns {missing} not in RefData "
                       f"(have: {list(df.columns)})")

    import pandas as pd
    label_col = cfg.classification.get("label_column", "New Problem Type")
    docs = []
    for idx, row in df[cols].iterrows():
        # Skip empty cells instead of emitting "Data_address: nan": only 18 of
        # the 96 reference rows have a Data_address, and a literal "nan" is
        # noise in the retrieval text as well as wasted prompt tokens.
        parts = []
        for c in cols:
            v = row[c]
            if pd.isna(v) or str(v).strip() in ("", "nan"):
                continue
            parts.append(f"{c}: {str(v).strip()}")
        text = "\n".join(parts)
        meta = {"row": int(idx)}
        if label_col in df.columns:
            meta["problem_type"] = str(df.at[idx, label_col])
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def refdata_token_report(cfg: ExpConfig, encoding: str = "o200k_base"):
    """
    How many prompt tokens the column selection saves, per classification call.
    Printed once so the number can go straight into the paper.
    """
    df = load_refdata(cfg)
    cols = cfg.classification.get("refdata_columns",
                                  ["prompt", "New Problem Type"])
    k = cfg.k("refdata")

    import pandas as pd

    def row_text(r, columns):
        return "\n".join(
            f"{c}: {str(r[c]).strip()}" for c in columns
            if not pd.isna(r[c]) and str(r[c]).strip() not in ("", "nan"))

    def row_tokens(columns):
        return [_estimate_tokens(row_text(r, columns), encoding)
                for _, r in df.iterrows()]

    full = row_tokens(list(df.columns))
    slim = row_tokens(cols)
    out = {
        "n_rows": len(df),
        "all_columns": list(df.columns),
        "kept_columns": cols,
        "k": k,
        "mean_tokens_per_doc_before": round(sum(full) / len(full), 1),
        "mean_tokens_per_doc_after": round(sum(slim) / len(slim), 1),
    }
    out["retrieved_tokens_before"] = round(out["mean_tokens_per_doc_before"] * k)
    out["retrieved_tokens_after"] = round(out["mean_tokens_per_doc_after"] * k)
    out["saved_per_call"] = (out["retrieved_tokens_before"]
                             - out["retrieved_tokens_after"])
    out["reduction_pct"] = round(
        100 * out["saved_per_call"] / max(1, out["retrieved_tokens_before"]), 1)
    return out


PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,                      # local server, no key
    "huggingface": None,                 # local sentence-transformers, no key
}


def ensure_api_keys(cfg: ExpConfig, interactive: bool = True) -> List[str]:
    """
    Make sure the API keys required by THIS profile are present.

    Lookup order:
      1. already in os.environ (including anything a .env file put there)
      2. interactive getpass prompt (VS Code / Jupyter show an input box)

    Only the providers this profile actually uses are checked, so the
    gpt-oss-20b runs never ask for an OpenAI key. Keys are never written to
    disk and never appear in the run log.
    """
    needed, missing = [], []
    for spec in cfg.models.values():
        if PROVIDER_ENV.get(spec.provider) is None:
            continue                      # local server, no key
        if spec.api_key_env not in needed:
            needed.append(spec.api_key_env)
    if PROVIDER_ENV.get(cfg.embedding_provider) is not None:
        if cfg.embedding_api_key_env not in needed:
            needed.append(cfg.embedding_api_key_env)

    def looks_real(v: str) -> bool:
        # "sk-...", "sk-or-v1-...", an empty string: all placeholders copied
        # straight out of .env.example. Catching them here turns a confusing
        # 401 from the provider into a clear message.
        v = (v or "").strip()
        return len(v) >= 20 and not v.endswith("...")

    for var in needed:
        cur = os.environ.get(var, "")
        if looks_real(cur):
            continue
        if cur:
            print(f"[leanopt_exp] {var} is set to a placeholder "
                  f"({cur[:12]}...), treating it as missing")
        if interactive and sys.stdin is not None:
            try:
                import getpass
                val = getpass.getpass(f"{var} (input hidden): ").strip()
                if val:
                    os.environ[var] = val
                    continue
            except Exception:
                pass
        missing.append(var)

    if missing:
        raise RuntimeError(
            f"missing API key(s): {', '.join(missing)}. Put them in a .env "
            f"file at the repository root (see .env.example) or export them "
            f"before starting the kernel."
        )

    if needed:
        shown = ", ".join(f"{v}=***{os.environ[v][-4:]}" for v in needed)
        print(f"[leanopt_exp] credentials ready: {shown}")
    else:
        print("[leanopt_exp] local models only; no API key needed")
    return needed


def build_llm(cfg: ExpConfig, role: str, **overrides):
    """
    Single entry point replacing every `ChatOpenAI(...)` / `ChatOllama(...)`
    in the notebooks. The tracker is attached here, once.
    """
    spec = cfg.spec(role)
    TRACKER.bind(cfg)
    common = dict(temperature=spec.temperature, callbacks=[TRACKER])
    if spec.max_tokens is not None:
        common["max_tokens"] = spec.max_tokens

    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI
        kw = dict(model=spec.model, top_p=spec.top_p, n=spec.n,
                  api_key=os.environ.get(spec.api_key_env), **common)
        if spec.seed is not None:
            kw["seed"] = spec.seed
        if spec.base_url:
            kw["base_url"] = spec.base_url
        kw.update(spec.extra)
        kw.update(overrides)
        return ChatOpenAI(**kw)

    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama
        kw = dict(model=spec.model,
                  base_url=spec.base_url or "http://localhost:11434",
                  top_p=spec.top_p, **common)
        if spec.seed is not None:
            kw["seed"] = spec.seed        # Ollama: deterministic sampling
        if spec.max_tokens is not None:
            kw.pop("max_tokens", None)
            kw["num_predict"] = spec.max_tokens
        kw.update(spec.extra)
        kw.update(overrides)
        return ChatOllama(**kw)

    if spec.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kw = dict(model=spec.model, top_p=spec.top_p, **common)
        kw.update(spec.extra); kw.update(overrides)
        return ChatGoogleGenerativeAI(**kw)

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kw = dict(model=spec.model, top_p=spec.top_p, **common)
        kw.update(spec.extra); kw.update(overrides)
        return ChatAnthropic(**kw)

    raise ValueError(f"unknown provider: {spec.provider}")


try:  # pragma: no cover
    from langchain_core.embeddings import Embeddings as _EmbeddingsBase
except Exception:  # pragma: no cover
    class _EmbeddingsBase:            # offline fallback for test_smoke.py
        pass


class CountingEmbeddings(_EmbeddingsBase):
    """
    Wrapper that accounts for embedding calls.

    MUST subclass langchain_core Embeddings: FAISS checks
    `isinstance(embedding, Embeddings)` and, when the check fails, falls back
    to treating the object as a plain function and calling it directly. That
    path only triggers at query time (not when the index is built), which is
    why building an index appears to work and the first retrieval then fails
    with "'CountingEmbeddings' object is not callable".

    LangChain embeddings do not emit callback events, so every
    FAISS.from_documents(...) was spending money outside the run log. The unit
    price is two orders of magnitude below the chat models, but "we measured
    every API call" has to be literally true, and for the k=1000 stores the row
    counts are not trivial.

    Token counts are tiktoken estimates (the embeddings endpoint does not
    return usage), and are flagged as such in the log.
    """

    def __init__(self, inner, cfg: ExpConfig, price_per_1m: float):
        self._inner = inner
        self._cfg = cfg
        self._price = price_per_1m

    def __getattr__(self, name):            # delegate everything else
        return getattr(self._inner, name)

    def _record(self, texts: List[str], kind: str, t0: float):
        rec = _CURRENT_RECORD.get()
        if rec is None:
            return
        enc = self._cfg.token_fallback_encoder
        tok = sum(_estimate_tokens(t, enc) for t in texts)
        rec.add_call({
            "type": "embedding",
            "stage": _CURRENT_STAGE.get(),
            "model": self._cfg.embedding_model,
            "n_texts": len(texts),
            "prompt_tokens": tok,
            "completion_tokens": 0,
            "total_tokens": tok,
            "token_source": "estimated",     # endpoint returns no usage
            "latency_s": time.perf_counter() - t0,
            "cost_usd": (tok / 1e6) * self._price,
            "ok": True,
            "call": kind,
        })

    def embed_documents(self, texts, *a, **kw):
        t0 = time.perf_counter()
        out = self._inner.embed_documents(texts, *a, **kw)
        self._record(list(texts), "embed_documents", t0)
        return out

    def embed_query(self, text, *a, **kw):
        t0 = time.perf_counter()
        out = self._inner.embed_query(text, *a, **kw)
        self._record([text], "embed_query", t0)
        return out

    # async variants: some chains use them, and the base class only provides
    # defaults that delegate to the sync methods via a thread pool.
    async def aembed_documents(self, texts, *a, **kw):
        t0 = time.perf_counter()
        out = await self._inner.aembed_documents(texts, *a, **kw)
        self._record(list(texts), "aembed_documents", t0)
        return out

    async def aembed_query(self, text, *a, **kw):
        t0 = time.perf_counter()
        out = await self._inner.aembed_query(text, *a, **kw)
        self._record([text], "aembed_query", t0)
        return out

    def __call__(self, text):
        """
        Last-resort compatibility: older FAISS code paths call the embedding
        object directly. Behaves like embed_query so nothing breaks even if a
        library takes that route.
        """
        return self.embed_query(text)


def build_embeddings(cfg: ExpConfig, count: bool = True):
    emb = _build_embeddings_raw(cfg)
    if count and cfg.embedding_price_per_1m:
        return CountingEmbeddings(emb, cfg, cfg.embedding_price_per_1m)
    return emb


def _build_embeddings_raw(cfg: ExpConfig):
    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kw = dict(model=cfg.embedding_model,
                  api_key=os.environ.get(cfg.embedding_api_key_env))
        if cfg.embedding_base_url:
            kw["base_url"] = cfg.embedding_base_url
            # OpenAI-compatible gateways reject the tokenised batch format
            kw["check_embedding_ctx_length"] = False
        return OpenAIEmbeddings(**kw)

    if cfg.embedding_provider == "huggingface":
        # fully local, no server and no key: pip install langchain-huggingface
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    if cfg.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.embedding_model,
            base_url=cfg.embedding_base_url or "http://localhost:11434")
    raise ValueError(f"unknown embedding provider: {cfg.embedding_provider}")


def build_retriever(cfg: ExpConfig, vectorstore, name: str = "default",
                    search_kwargs: Optional[dict] = None, **overrides):
    """
    Centralises k / max_tokens_limit / filters so retrieval settings are
    reported in one place instead of being scattered over 20 notebook cells.

    The numbers come from cfg.retrievers[name]; they are the values the
    notebooks already used (k ranges from 1 to 1000 depending on the call
    site), so behaviour is unchanged -- only the bookkeeping moves.
    """
    rc = cfg.retriever_cfg(name)
    sk = {"k": rc["k"]}
    if search_kwargs:
        sk.update(search_kwargs)
    kw = {"search_type": rc.get("search_type", "similarity"), "search_kwargs": sk}
    if rc.get("max_tokens_limit") is not None:
        kw["max_tokens_limit"] = rc["max_tokens_limit"]
    kw.update(overrides)
    return vectorstore.as_retriever(**kw)


def agent_kwargs(cfg: ExpConfig, prefix: str, suffix: str,
                 input_variables: Optional[List[str]] = None) -> dict:
    """Uniform initialize_agent(**agent_kwargs(...)) settings for all agents."""
    ak = {"prefix": prefix, "suffix": suffix}
    if input_variables:
        ak["input_variables"] = input_variables
    return {
        "agent_kwargs": ak,
        "verbose": cfg.verbose,
        "handle_parsing_errors": cfg.handle_parsing_errors,
        "max_iterations": cfg.agent_max_iterations,
        "max_execution_time": cfg.agent_max_execution_time,
        "early_stopping_method": cfg.agent_early_stopping_method,
        "callbacks": [TRACKER],
        "return_intermediate_steps": False,
    }


# Errors that will never succeed on retry: no credit, bad key, no permission.
# Retrying these just burns wall-clock (and, across 101 instances, a lot of it).
_FATAL_ERROR_MARKERS = (
    "insufficient_quota",          # 429 but really "your balance is empty"
    "invalid_api_key",
    "authentication",
    "permission_denied",
    "model_not_found",
    "billing_not_active",
)


def is_fatal_api_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _FATAL_ERROR_MARKERS)


def call_with_retry(fn, cfg: ExpConfig, *args, **kwargs):
    """Uniform failure handling; retries are counted into the current record."""
    last = None
    for attempt in range(cfg.max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:                       # noqa: BLE001
            last = e
            rec = _CURRENT_RECORD.get()
            if rec is not None:
                rec.n_retries += 1
                rec.errors.append(f"attempt{attempt}: {type(e).__name__}: {e}")
            if is_fatal_api_error(e):
                print(f"[leanopt_exp] not retrying, this will not fix itself: "
                      f"{type(e).__name__}: {str(e)[:160]}")
                break
            if attempt == cfg.max_retries - 1:
                break
            time.sleep(cfg.retry_backoff_s * (2 ** attempt))
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 3.  Run logging
# --------------------------------------------------------------------------- #

@dataclass
class InstanceRecord:
    run_id: str
    instance_id: Any
    dataset: str
    method: str
    model_profile: str
    repeat_index: int
    query: str = ""
    gold_type: Optional[str] = None
    pred_type: Optional[str] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    n_retries: int = 0
    n_provider_retries: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    t_start: float = 0.0
    wall_s: Optional[float] = None
    status: str = "pending"

    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def add_call(self, call: Dict[str, Any]):
        call["t_rel_s"] = round(time.perf_counter() - self.t_start, 4)
        with self._lock:
            self.calls.append(call)

    def add_trace(self, item: Dict[str, Any]):
        with self._lock:
            self.trace.append(item)

    def fail(self, exc: BaseException, note: str = ""):
        """
        Mark this instance as failed when the exception is caught inside the
        `with log.instance(...)` block. Without this the logger sees a clean
        exit and records status='ok', which would understate the failure rate.
        """
        self.status = "failed"
        msg = f"{type(exc).__name__}: {exc}"
        self.errors.append(f"{msg} ({note})" if note else msg)

    def set(self, **kwargs):
        """Attach outputs / labels: rec.set(pred_type=..., code_output=...)."""
        for k, v in kwargs.items():
            if k in {f.name for f in dataclasses.fields(InstanceRecord)}:
                setattr(self, k, v)
            else:
                self.extra[k] = v

    # ---- aggregation ---------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        llm = [c for c in self.calls if c.get("type") == "llm"]
        tools = [c for c in self.calls if c.get("type") == "tool"]
        retr = [c for c in self.calls if c.get("type") == "retriever"]
        embs = [c for c in self.calls if c.get("type") == "embedding"]
        def _blank():
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cost_usd": 0.0, "latency_s": 0.0,
                    "tool_calls": 0, "retriever_calls": 0, "tool_latency_s": 0.0}

        by_stage: Dict[str, Dict[str, float]] = {}
        for c in llm:
            s = by_stage.setdefault(c.get("stage", "unassigned"), _blank())
            s["calls"] += 1
            s["prompt_tokens"] += c.get("prompt_tokens") or 0
            s["completion_tokens"] += c.get("completion_tokens") or 0
            s["cost_usd"] += c.get("cost_usd") or 0.0
            s["latency_s"] += c.get("latency_s") or 0.0
        for c in tools:
            s = by_stage.setdefault(c.get("stage", "unassigned"), _blank())
            s["tool_calls"] += 1
            s["tool_latency_s"] += c.get("latency_s") or 0.0
        for c in retr:
            s = by_stage.setdefault(c.get("stage", "unassigned"), _blank())
            s["retriever_calls"] += 1
        return {
            "n_llm_calls": len(llm),
            "n_llm_calls_failed": sum(1 for c in llm if not c.get("ok", True)),
            "n_tool_calls": len(tools),
            "n_tool_calls_failed": sum(1 for c in tools if not c.get("ok", True)),
            "n_retriever_calls": len(retr),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in llm),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in llm),
            "cached_prompt_tokens": sum(c.get("cached_prompt_tokens") or 0
                                        for c in llm),
            "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in llm),
            "total_tokens": sum(c.get("total_tokens") or 0 for c in llm),
            "n_embedding_calls": len(embs),
            "embedding_tokens": sum(c.get("prompt_tokens") or 0 for c in embs),
            "embedding_cost_usd": round(
                sum(c.get("cost_usd") or 0.0 for c in embs), 6),
            "cost_usd": round(sum(c.get("cost_usd") or 0.0
                                  for c in llm + embs), 6),
            "llm_cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in llm), 6),
            "llm_latency_s": round(sum(c.get("latency_s") or 0.0 for c in llm), 3),
            "tool_latency_s": round(sum(c.get("latency_s") or 0.0 for c in tools), 3),
            "wall_s": self.wall_s,
            "n_estimated_token_calls": sum(
                1 for c in llm if c.get("token_source") == "estimated"),
            "n_price_missing_calls": sum(
                1 for c in llm if c.get("price_missing")),
            "by_stage": by_stage,
        }

    def to_json(self, keep_prompts: bool = True) -> Dict[str, Any]:
        calls = self.calls
        if not keep_prompts:
            calls = [{k: v for k, v in c.items()
                      if k not in ("prompts", "completion")} for c in calls]
        return {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "dataset": self.dataset,
            "method": self.method,
            "model_profile": self.model_profile,
            "repeat_index": self.repeat_index,
            "status": self.status,
            "query": self.query,
            "gold_type": self.gold_type,
            "pred_type": self.pred_type,
            "n_retries": self.n_retries,
            "n_provider_retries": self.n_provider_retries,
            "errors": self.errors,
            "summary": self.summary(),
            "calls": calls,
            "trace": self.trace,
            **self.extra,
        }


class BudgetExceeded(RuntimeError):
    """Raised after an instance pushes the run over cfg.budget_usd_per_run."""


class RunLogger:
    """
    Writes:
      runs/<run_id>/config.json     -- full ExpConfig + environment manifest
      runs/<run_id>/instances.jsonl -- one line per benchmark instance
      runs/<run_id>/calls.jsonl     -- one line per LLM/tool call (flat, for plots)
      runs/<run_id>/summary.json    -- run-level aggregate (written on close)
    """

    def __init__(self, cfg: ExpConfig, out_dir: Optional[str] = None):
        self.cfg = cfg
        TRACKER.bind(cfg)
        # The directory is NOT created here. Re-running the configuration cell
        # while debugging would otherwise litter runs/ with empty directories.
        # Everything is created on first use -- i.e. the first LOG.instance()
        # or the first access to LOG.dir.
        self._dir = Path(out_dir or cfg.out_dir) / cfg.run_id
        self._started = False
        self._inst_f = None
        self._call_f = None
        self.records: List[InstanceRecord] = []
        print(f"[leanopt_exp] run_id = {cfg.run_id}")
        # Show the dataset up front. Stored in config.json it is only auditable
        # after the fact; printed here it is checkable before any money is
        # spent -- "101 rows" vs "14 rows" is the cheapest possible way to
        # notice you are not running what you think you are.
        ds = (cfg.manifest or {}).get("dataset") or {}
        if ds.get("present"):
            rows = ds.get("rows")
            print(f"[leanopt_exp] dataset  = {ds['path']}"
                  f"{f' ({rows} rows)' if rows is not None else ''}"
                  f"  sha256:{ds['sha256'][:12]}")
        elif ds.get("path"):
            print(f"[leanopt_exp] dataset  = {ds['path']}  !! file not found")
        print(f"[leanopt_exp] will log to {self._dir} (created on first result)")

    # ------------------------------------------------------------------ #
    @property
    def dir(self) -> Path:
        """Run directory; created on first access."""
        self._ensure()
        return self._dir

    @property
    def dir_path(self) -> Path:
        """Where the run *would* be written, without creating anything."""
        return self._dir

    def _ensure(self):
        if self._started:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8")
        self._inst_f = (self._dir / "instances.jsonl").open("a", encoding="utf-8")
        self._call_f = (self._dir / "calls.jsonl").open("a", encoding="utf-8")
        self._started = True
        print(f"[leanopt_exp] logging to {self._dir.resolve()}")

    @contextlib.contextmanager
    def instance(self, instance_id, query: str = "", gold_type=None, **extra):
        rec = InstanceRecord(
            run_id=self.cfg.run_id, instance_id=instance_id,
            dataset=self.cfg.dataset, method=self.cfg.method,
            model_profile=self.cfg.model_profile,
            repeat_index=self.cfg.repeat_index,
            query=query, gold_type=gold_type,
        )
        rec.extra.update(extra)
        rec.t_start = time.perf_counter()
        token = _CURRENT_RECORD.set(rec)
        try:
            yield rec
            # don't overwrite a status already set by rec.fail() inside the
            # block -- a caught exception is still a failed instance
            if rec.status == "pending":
                rec.status = "ok"
        except BaseException as e:                    # noqa: BLE001
            # BaseException, not Exception: a KeyboardInterrupt (stopping the
            # cell in Jupyter) used to slip through and leave the row stuck at
            # status="pending", which then polluted the aggregation.
            rec.status = ("interrupted"
                          if isinstance(e, KeyboardInterrupt) else "failed")
            rec.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            _CURRENT_RECORD.reset(token)
            rec.wall_s = round(time.perf_counter() - rec.t_start, 3)
            self._flush(rec)
            if self.cfg.sleep_between_instances_s:
                time.sleep(self.cfg.sleep_between_instances_s)
        self.check_budget()

    def _flush(self, rec: InstanceRecord):
        self._ensure()
        self.records.append(rec)
        self._inst_f.write(json.dumps(rec.to_json(self.cfg.log_prompts),
                                      default=str) + "\n")
        self._inst_f.flush()
        for c in rec.calls:
            flat = {k: v for k, v in c.items() if k not in ("prompts", "completion")}
            flat.update({"run_id": rec.run_id, "instance_id": rec.instance_id,
                         "method": rec.method, "dataset": rec.dataset,
                         "model_profile": rec.model_profile})
            self._call_f.write(json.dumps(flat, default=str) + "\n")
        self._call_f.flush()

    # ------------------------------------------------------------------ #
    def spent_usd(self) -> float:
        return sum(r.summary()["cost_usd"] for r in self.records)

    def done_instance_ids(self) -> set:
        """
        Instance ids already present in this run directory. Lets a batch be
        resumed after an interruption without paying for the same instances
        twice. Only successful instances count.
        """
        f = self._dir / "instances.jsonl"
        if not f.exists():
            return set()
        out = set()
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("status") == "ok":
                out.add(d.get("instance_id"))
        return out

    def check_budget(self):
        cap = self.cfg.budget_usd_per_run
        if cap and self.spent_usd() > cap:
            raise BudgetExceeded(
                f"run has spent ${self.spent_usd():.4f}, over the "
                f"${cap:.2f} budget set in exp_config.yaml "
                f"(budget_usd_per_run). {len(self.records)} instances done.")

    def close(self) -> Dict[str, Any]:
        if not self._started:
            print("[leanopt_exp] no instances recorded; nothing written")
            return {"run_id": self.cfg.run_id, "n_instances": 0}
        agg = {"run_id": self.cfg.run_id, "method": self.cfg.method,
               "dataset": self.cfg.dataset, "model_profile": self.cfg.model_profile,
               "n_instances": len(self.records)}
        keys = ["n_llm_calls", "n_tool_calls", "n_retriever_calls",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "cost_usd", "llm_latency_s", "wall_s"]
        for k in keys:
            vals = [r.summary().get(k) or 0 for r in self.records]
            agg[f"total_{k}"] = round(sum(vals), 6)
            agg[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0
        agg["n_failed"] = sum(1 for r in self.records if r.status == "failed")
        (self._dir / "summary.json").write_text(
            json.dumps(agg, indent=2, default=str), encoding="utf-8")
        self._inst_f.close()
        self._call_f.close()
        # Safe to call close() after each slice of a batch: the next instance
        # simply reopens the same files in append mode.
        self._started = False
        print(f"[leanopt_exp] {agg['n_instances']} instances | "
              f"{agg['total_n_llm_calls']} LLM calls | "
              f"{agg['total_total_tokens']:,} tokens | "
              f"${agg['total_cost_usd']:.4f} | "
              f"{agg['total_wall_s']:.1f}s")
        return agg
