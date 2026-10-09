# CELL 2
from __future__ import annotations
import ast
import fcntl
import gurobipy as gp
import hashlib
import json
import math
import numpy as np
import os
import openai
import pandas as pd
import pprint
import re
import tempfile
from IPython.display import display
from dotenv import load_dotenv
from functools import lru_cache
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import quote

import httpx


# CELL 5
HTTP_RETRY_EVENTS = []


def _record_http_request(request):
    retry_index = int(request.headers.get("x-stainless-retry-count", "0"))
    if retry_index:
        HTTP_RETRY_EVENTS.append({"endpoint": request.url.path, "retry_index": retry_index})


@lru_cache(maxsize=1)
def _get_http_client():
    return httpx.Client(timeout=180, event_hooks={"request": [_record_http_request]})


class ModelOutputTruncated(RuntimeError):
    """The API explicitly reported an incomplete model response."""


class RejectTruncatedOutput(BaseCallbackHandler):
    raise_error = True

    def on_llm_end(self, response, **kwargs):
        for choices in response.generations:
            for generation in choices:
                metadata = getattr(getattr(generation, "message", None), "response_metadata", {}) or {}
                info = generation.generation_info or {}
                reason = str(info.get("finish_reason") or metadata.get("finish_reason") or "").lower()
                if reason in {"length", "max_tokens", "max_output_tokens"}:
                    error = ModelOutputTruncated(f"Model output is incomplete: finish_reason={reason}")
                    error.llm_usage = (response.llm_output or {}).get("token_usage", {})
                    raise error


def require_api_key():
    """Load credentials at use time, including after configuring a running kernel."""
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = (os.environ.get("OPENAI_API_KEY") or user_api_key or "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in the environment or in "
            f"{PROJECT_ROOT / '.env.local'} before running a test."
        )
    return key


def make_llm(*, timeout=180):
    return ChatOpenAI(
        model_name=MODEL_SNAPSHOT, openai_api_key=require_api_key(),
        temperature=0.0, top_p=1, n=1, max_retries=API_MAX_RETRIES, http_client=_get_http_client(), timeout=timeout,
        callbacks=[RejectTruncatedOutput()],
    )


def make_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=require_api_key(), max_retries=API_MAX_RETRIES, http_client=_get_http_client())

MODEL_INTERFACE_INSTRUCTIONS = """
Mandatory execution interface, independent of every retrieved example:
Create exactly one Gurobi model and expose the solved, live model at module scope as m.
Preferred function pattern (replace the placeholder with the actual modeling code):
    def solve_problem(...):
        m = gp.Model(...)
        # define variables, objective, and all constraints
        m.optimize()
        return m
    m = solve_problem(...)
A direct global script is also allowed, but its final m must be the valid Gurobi model.
If a function creates the model, explicitly return m and call that function at module scope.
Defining a function without calling it is incomplete. Reserve m and model for Gurobi models.
Do not assign the return of optimize() to m/model: never write m = m.optimize().
Do not close or dispose the model, or exit a model context manager, before the outer harness
extracts results. Do not catch and hide execution errors. The harness checks the actual solver
status and extracts objective and variable values; printed values cannot replace this interface.
"""


# CELL 7
def normalize_route(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "NETWORK REVENUE MANAGEMENT": "NRM", "NETWORK REVENUE MANAGEMENT PROBLEM": "NRM",
        "RESOURCE ALLOCATION": "RA", "RESOURCE ALLOCATION PROBLEM": "RA",
        "TRANSPORTATION": "TP", "TRANSPORTATION PROBLEM": "TP",
        "ASSIGNMENT": "AP", "ASSIGNMENT PROBLEM": "AP",
        "FACILITY LOCATION": "FLP", "FACILITY LOCATION PROBLEM": "FLP",
    }
    text = aliases.get(text, text)
    if text in {"MIXTURE", "OTHER", "OTHERS"} or text.startswith("OTHERS"):
        return "Others"
    if text == "UFLP":
        return "FLP"
    if text in {"NRM", "RA", "TP", "AP", "FLP"}:
        return text
    raise ValueError(f"Unsupported execution route: {value!r}")


FILTER_OPERATORS = {
    "exact", "prefix", "contains", "in", "not_in", "eq", "ne",
    "gt", "ge", "lt", "le", "between", "is_null", "not_null",
}


FILTER_DTYPES = {"string", "number", "date"}


CSVQA_LEGACY_SYSTEM_PROMPT = (
    "Get every CSV row needed by the query in source order. Preserve exact identifiers and "
    "numeric values. If the query names a subset, get that complete subset; otherwise get "
    "all data. Use only the supplied context: {context}"
)


CSVQA_LEGACY_TOOL_DESCRIPTION = (
    "Get complete source-ordered CSV evidence needed to formulate the optimization problem."
)

REACT_PROTOCOL_EVENTS = []
REACT_PROTOCOL_DEADLINE = None


class ReActProtocolError(RuntimeError):
    """A discarded protocol attempt, not a completed benchmark outcome."""


def react_protocol_record_fields():
    return {"retry_count": len(REACT_PROTOCOL_EVENTS),
            "protocol_retry_count": len(REACT_PROTOCOL_EVENTS),
            "protocol_retry_events": json.dumps(REACT_PROTOCOL_EVENTS, ensure_ascii=False)}


def invoke_react_protocol(agent_factory, query, route, *, required_tool=None):
    """Restart only malformed ReAct output or a missing required CSVQA call."""
    import time
    from langchain_core.exceptions import OutputParserException
    deadline = REACT_PROTOCOL_DEADLINE or (time.monotonic() + REACT_PROTOCOL_TIMEOUT_SECONDS)
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("ReAct protocol remained incomplete at the common case deadline")
        try:
            agent = agent_factory()
            original_template = agent.agent.llm_chain.prompt.template
            if REACT_PROTOCOL_EVENTS:
                prompt = agent.agent.llm_chain.prompt
                marker = "PROTOCOL RESTART REMINDER:"
                if marker not in prompt.template:
                    prompt.template += ("\n" + marker + " Use exact Thought/Action/Action Input or "
                                        "Thought/Final Answer labels. Do not output a bare model." +
                                        (" Call CSVQA before the Final Answer." if required_tool else ""))
            try:
                result = agent.invoke(query)
            finally:
                agent.agent.llm_chain.prompt.template = original_template
            output = str(result.get("output", "") or "").strip()
            calls = sum(getattr(step[0], "tool", None) == required_tool
                        for step in result.get("intermediate_steps", [])
                        if isinstance(step, tuple) and step) if required_tool else 0
            if required_tool and calls < 1:
                raise ReActProtocolError("missing_required_csvqa")
            if not output or "agent stopped due to" in output.lower():
                raise ReActProtocolError("incomplete_final_answer")
            return result
        except (OutputParserException, ValueError, ReActProtocolError) as exc:
            cause, parser_error = exc, False
            while cause is not None:
                parser_error = parser_error or isinstance(cause, OutputParserException)
                cause = cause.__cause__
            if not parser_error and not isinstance(exc, ReActProtocolError):
                raise
            reason = "react_output_format" if parser_error else str(exc)
            REACT_PROTOCOL_EVENTS.append({"route": str(route), "reason": reason,
                                          "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            print(f"[ReAct protocol restart] {route}: {reason}; discard this attempt and restart the agent.")


def invoke_react_with_required_csvqa(agent_factory, query, route):
    """Return the first valid ReAct result with at least one actual CSVQA call."""
    return invoke_react_protocol(agent_factory, query, route, required_tool="CSVQA")


CSVQA_PLANNED_PROMPTS = {'NRM': 'Select every source row and column needed to define the revenue-management objective, decision entities, demand limits, inventory or capacity limits, and query-specific bounds. Infer exact columns from the query and schema. Filter only an explicit subset.'}

CSVQA_PLANNED_PROMPTS["RA"] = (
    "Select source columns for decision identifiers, objective coefficients, resource consumption, "
    "every applicable capacity dimension, and query-specific bounds. Keep required scalar parameter "
    "rows and resource tables even when products are filtered. Infer roles from the query and schema, "
    "never column position, numeric magnitude, or filename alone. Exclude administrative/redundant "
    "columns only when they have no role in the query. Preserve exact IDs and all required coefficients. "
    "Filter only subsets explicitly requested by the original query."
)

CSVQA_TOOL_DESCRIPTIONS = {'NRM': 'Return the canonical CSV rows, columns, filters, and relationships needed by the query.'}

CSVQA_TOOL_DESCRIPTIONS["RA"] = "Return source-derived RA records, columns, filters, and resource relationships."


def read_csv_compat(path, **kwargs):
    """Read a project-relative or absolute CSV with the requested pandas options."""
    path = Path(path).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(f"Unable to decode CSV: {path}") from last_error


def _read_csv(path):
    """Read values as text and remove only empty structure or a clear export index."""
    frame = read_csv_compat(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    if frame.empty:
        return frame
    nonempty = frame.apply(lambda column: column.astype(str).str.strip().ne(""))
    frame = frame.loc[nonempty.any(axis=1), nonempty.any(axis=0)].copy()
    for column in list(frame.columns):
        values = frame[column].astype(str).tolist()
        sequential = values in (
            [str(i) for i in range(len(values))],
            [str(i) for i in range(1, len(values) + 1)],
        )
        if sequential and re.fullmatch(r"(?:unnamed(?::\s*\d+)?|index)", column.strip(), re.I):
            frame = frame.drop(columns=column)
    return frame


def _load_tables(dataset_address, file_indices=None):
    """Read requested CSV files, retaining their original zero-based source indices."""
    paths = [raw.strip() for raw in str(dataset_address or "").splitlines() if raw.strip()]
    if file_indices is not None:
        if (not isinstance(file_indices, list) or not file_indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(paths)
                       for i in file_indices)):
            raise ValueError("CSVQA file_indices must be nonempty valid zero-based source indices")
    tables = []
    for index, raw in enumerate(paths):
        if file_indices is not None and index not in file_indices:
            continue
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSVQA cannot find dataset file: {path}")
        tables.append({"file_index": index, "path": path, "frame": _read_csv(path)})
    if not tables:
        raise ValueError("CSVQA requires at least one CSV path")
    return tables


def _build_profile(tables, query):
    """Expose schema, compact samples, and evidence for query-named terms."""
    quote_pattern = r'"([^\"]+)"|\'([^\']+)\'|“([^”]+)”|‘([^’]+)’'
    terms = list(dict.fromkeys(
        next(value for value in match.groups() if value is not None).strip()
        for match in re.finditer(quote_pattern, str(query or ""))
    ))
    key = lambda value: "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
    profile = []
    for table in tables:
        frame = table["frame"]
        term_matches = {}
        for term in terms:
            wanted, value_matches = term.casefold(), {}
            for column in frame.columns:
                values = frame[column].astype(str).str.strip().str.casefold()
                counts = {
                    "exact": int(values.eq(wanted).sum()),
                    "prefix": int(values.str.startswith(wanted).sum()),
                    "contains": int(values.str.contains(wanted, regex=False).sum()),
                }
                if counts["contains"]:
                    value_matches[str(column)] = counts
            term_matches[term] = {
                "column_matches": [str(column) for column in frame.columns if key(column) == key(term)],
                "value_matches": value_matches,
            }
        profile.append({
            "file_index": table["file_index"],
            "file_name": table["path"].name,
            "rows": len(frame),
            "columns": list(frame.columns),
            "sample_values": {
                str(column): frame[column].astype(str).loc[lambda values: values.str.strip().ne("")]
                .drop_duplicates().head(4).tolist()
                for column in frame.columns
            },
            "query_terms": term_matches,
        })
    return profile


for _route in ("TP", "AP", "FLP"):
    CSVQA_PLANNED_PROMPTS[_route] = (
        "Select every source row and column required by the original query, including entity IDs, "
        "matrix coefficients and axes, costs, capacities, demands and additional restrictions. "
        "Use exact source column names and retain business keys. Exclude a column only when it "
        "has no role in the query. Illustrative examples do not restrict the entity set."
    )
    CSVQA_TOOL_DESCRIPTIONS[_route] = "Return source-derived records, exact columns, filters and matrix axes."

def source_csvqa_frames(payload):
    """Create source-only DataFrames keyed by exact table_id; retain strings and row order."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    return {table["table_id"]: pd.DataFrame(
        [record["values"] for record in table["records"]], columns=table["columns"],
        index=[record["source_row"] for record in table["records"]],
    ) for table in data["tables"]}


# CELL 9
def _ask_extraction_plan(llm, route, query, tables, route_prompt):
    """Ask for one declarative plan and return the parsed plan plus raw response."""
    profile = json.dumps(_build_profile(tables, query), ensure_ascii=False)
    request = f"""You are a CSV extraction planner. Return one JSON object only.
Choose tables, columns, and filters from the user query and the supplied CSV profile. Do not return
data, Python, formulas, prose, joins, aggregation, sorting, sampling, or invented fields.

Shared plan schema:
{{"route":"{route}","tables":[{{"file_index":0,"role":"descriptive non-unique role",
"columns":["exact column"] or "*","filters":{{"logic":"and|or","conditions":[
{{"column":"exact column","operator":"exact|prefix|contains|in|not_in|eq|ne|gt|ge|lt|le|between|is_null|not_null",
"dtype":"string|number|date","value":"scalar or list","evidence":"exact query span",
"format":"optional date format","inclusive":"both|left|right|neither"}}]}}}}],
"ignored_file_indices":[],"relationships":[{{"type":"matrix",
"matrix_table_id":"file_2_view_0","row_id_column":"exact column",
"row_axis":{{"table_id":"file_0_view_0","id_column":"exact column"}},
"column_axis":{{"table_id":"file_1_view_0","id_column":"exact column"}}}}]}}

Python assigns each logical view a table_id of file_<file_index>_view_<zero-based occurrence for that
file>. A physical file may appear in multiple table specifications and roles may repeat. Every input
file must appear in tables or ignored_file_indices. Ignore a file only when the query does not need it.
Filter only an explicit restriction; names introduced as examples (e.g., such as, for example)
do not restrict the entity set. Every filter must quote the query restriction that requires it.
Choose the filter operator from the requested matching semantics. A categorical code
classifying identifier families refers to a prefix when no separate category column exists;
arbitrary substring matching also admits unrelated identifiers and requires explicit query support.
Declare a matrix relationship only when the profile exposes its two axes unambiguously. When one label
column contains both decision entities and scalar parameter rows, retain every row that supplies a
query-required coefficient, price, cost, capacity, or bound; use separate logical views when helpful.

Route guidance:
{route_prompt}

User query:
{query}

CSV profile:
{profile}
"""
    text = str(llm.invoke([HumanMessage(content=request)]).content or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I) if text.startswith("```") else text
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        error = ValueError(f"CSVQA planner did not return a JSON object: {text[:200]}")
        error.planner_output = text
        raise error
    try:
        plan = json.loads(cleaned[start:end + 1])
    except ValueError as exc:
        error = ValueError(f"CSVQA planner returned invalid JSON: {exc}")
        error.planner_output = text
        raise error from exc
    if not isinstance(plan, dict):
        raise ValueError("CSVQA extraction plan must be a JSON object")
    return plan, text


def _apply_condition(series, condition):
    """Apply one allowlisted string, number, date, range, set, or null predicate."""
    operator, dtype = condition["operator"], condition.get("dtype", "string")
    raw, missing = series.astype(str).str.strip(), series.astype(str).str.strip().eq("")
    if operator in {"is_null", "not_null"}:
        return missing if operator == "is_null" else ~missing
    if dtype == "number":
        values, convert = pd.to_numeric(raw, errors="coerce"), float
    elif dtype == "date":
        date_format = condition.get("format") or None
        values = pd.to_datetime(raw, format=date_format, errors="coerce")
        convert = lambda value: pd.to_datetime(value, format=date_format)
    else:
        values = raw.str.casefold()
        convert = lambda value: str(value).strip().casefold()
    if dtype != "string" and values.loc[~missing].isna().any():
        raise ValueError(f"Cannot reliably parse {series.name!r} as {dtype}")
    value = condition.get("value")
    wanted = [convert(item) for item in value] if isinstance(value, list) else convert(value)
    if operator in {"in", "not_in", "between"} and not isinstance(wanted, list):
        raise ValueError(f"Operator {operator!r} requires a list value")
    if operator == "between" and len(wanted) != 2:
        raise ValueError("Operator 'between' requires exactly two bounds")
    operations = {
        "exact": lambda: values.eq(wanted), "eq": lambda: values.eq(wanted),
        "ne": lambda: values.ne(wanted) & ~missing,
        "prefix": lambda: values.str.startswith(wanted, na=False),
        "contains": lambda: values.str.contains(wanted, regex=False, na=False),
        "in": lambda: values.isin(wanted), "not_in": lambda: ~values.isin(wanted) & ~missing,
        "gt": lambda: values.gt(wanted), "ge": lambda: values.ge(wanted),
        "lt": lambda: values.lt(wanted), "le": lambda: values.le(wanted),
        "between": lambda: values.between(
            wanted[0], wanted[1], inclusive=condition.get("inclusive", "both")
        ),
    }
    if operator not in operations:
        raise ValueError(f"Unsupported non-null filter operator: {operator}")
    return operations[operator]()


def _execute_plan(plan, tables, query, route):
    """Validate and execute a query-driven plan; validate only declared relationships."""
    key = lambda value: "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
    route = normalize_route(route)
    if normalize_route(plan.get("route")) != route:
        raise ValueError(f"CSVQA plan route does not match {route}")
    specs = plan.get("tables")
    if not isinstance(specs, list) or not specs or any(not isinstance(spec, dict) for spec in specs):
        raise ValueError("CSVQA plan requires a non-empty tables list")
    ignored = plan.get("ignored_file_indices") or []
    if not isinstance(ignored, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in ignored):
        raise ValueError("ignored_file_indices must be a list of integers")
    by_index = {table["file_index"]: table for table in tables}
    used = {spec.get("file_index") for spec in specs}
    if used & set(ignored) or (used | set(ignored)) != set(by_index):
        raise ValueError("Every input file must be used or explicitly ignored exactly once")

    payload_tables, payload_by_id, used_conditions, view_counts = [], {}, [], {}
    for spec in specs:
        file_index, role = spec.get("file_index"), str(spec.get("role") or "data").strip()
        if isinstance(file_index, bool) or not isinstance(file_index, int) or file_index not in by_index:
            raise ValueError(f"Invalid CSV file_index: {file_index!r}")
        occurrence = view_counts.get(file_index, 0)
        view_counts[file_index] = occurrence + 1
        table_id = f"file_{file_index}_view_{occurrence}"
        table, frame = by_index[file_index], by_index[file_index]["frame"]
        requested = spec.get("columns", "*")
        columns = list(frame.columns) if requested == "*" or requested == ["*"] else requested
        if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Table {table_id!r} requires unique columns or '*'")
        unknown = [column for column in columns if column not in frame.columns]
        if unknown:
            raise ValueError(f"Unknown columns for table {table_id!r}: {unknown}")
        filters = spec.get("filters") or {}
        if not isinstance(filters, dict):
            raise ValueError(f"Filters for table {table_id!r} must be an object")
        logic, conditions = str(filters.get("logic", "and")).casefold(), filters.get("conditions") or []
        if not isinstance(conditions, list):
            raise ValueError(f"Invalid filter logic for table {table_id!r}")
        # A common schema slip puts the sole condition's operator in ``logic``.
        # It is unambiguous for one condition; multi-condition plans remain strict.
        if logic not in {"and", "or"}:
            sole_operator = str(conditions[0].get("operator", "")).casefold() if len(conditions) == 1 and isinstance(conditions[0], dict) else ""
            if logic != sole_operator:
                raise ValueError(f"Invalid filter logic for table {table_id!r}")
            logic = "and"
            filters = {**filters, "logic": logic}
        combined = None
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError(f"Invalid filter for table {table_id!r}: {condition!r}")
            column, operator = condition.get("column"), condition.get("operator")
            dtype, evidence = condition.get("dtype", "string"), str(condition.get("evidence") or "")
            if column not in frame.columns or operator not in FILTER_OPERATORS or dtype not in FILTER_DTYPES:
                raise ValueError(f"Invalid filter for table {table_id!r}: {condition}")
            raw_values = condition.get("value", [])
            raw_values = raw_values if isinstance(raw_values, list) else [raw_values]
            value_supported = bool(raw_values) and all(key(value) in key(query) for value in raw_values)
            if not evidence.strip() or (key(evidence) not in key(query) and not value_supported):
                raise ValueError(f"Filter lacks query evidence: {condition}")
            illustrative = re.search(r"\bsuch as\b|\bfor example\b|\be\.g\.", evidence, re.I)
            restrictive = re.search(r"\bonly\b|\bexclud\w*\b|\bexcept\b|\brestrict\w*\b|\blimited to\b", evidence, re.I)
            if illustrative and not restrictive:
                raise ValueError(f"Illustrative query examples cannot justify a source-row filter: {evidence}")
            mask = _apply_condition(frame[column], condition)
            combined = mask if combined is None else (combined & mask if logic == "and" else combined | mask)
            used_conditions.append(condition)
        selected = frame if combined is None else frame.loc[combined]
        if selected.empty:
            raise ValueError(f"CSVQA plan returned zero rows for table {table_id!r}")
        records = [
            {"source_row": int(index), "values": {column: str(row[column]) for column in columns}}
            for index, row in selected.iterrows()
        ]
        payload = {
            "table_id": table_id, "file_index": file_index, "file_name": table["path"].name,
            "role": role or "data", "columns": columns, "original_rows": len(frame),
            "returned_rows": len(records), "filters": filters, "records": records,
        }
        payload_tables.append(payload)
        payload_by_id[table_id] = payload

    profile = _build_profile(tables, query)
    required_columns = {
        column for file_profile in profile for matches in file_profile["query_terms"].values()
        for column in matches["column_matches"]
    }
    selected_columns = {column for table in payload_tables for column in table["columns"]}
    missing_columns = sorted(required_columns - selected_columns)
    if missing_columns:
        raise ValueError(f"Plan omitted columns named by the query: {missing_columns}")

    relationships = plan.get("relationships") or []
    if not isinstance(relationships, list):
        raise ValueError("relationships must be a list")
    validated_relationships, matrix_checks = [], []
    for relationship in relationships:
        if not isinstance(relationship, dict) or relationship.get("type") != "matrix":
            raise ValueError(f"Unsupported relationship: {relationship!r}")
        matrix_id = relationship.get("matrix_table_id")
        row_axis_spec, column_axis_spec = relationship.get("row_axis") or {}, relationship.get("column_axis") or {}
        if not isinstance(row_axis_spec, dict) or not isinstance(column_axis_spec, dict):
            raise ValueError("Matrix row_axis and column_axis must be objects")
        row_axis_id, column_axis_id = row_axis_spec.get("table_id"), column_axis_spec.get("table_id")
        if any(table_id not in payload_by_id for table_id in (matrix_id, row_axis_id, column_axis_id)):
            raise ValueError(f"Relationship references an unknown table_id: {relationship}")
        matrix_table, row_axis, column_axis = (
            payload_by_id[matrix_id], payload_by_id[row_axis_id], payload_by_id[column_axis_id]
        )
        row_id_column = relationship.get("row_id_column")
        row_axis_column, column_axis_column = row_axis_spec.get("id_column"), column_axis_spec.get("id_column")
        for table, column in (
            (matrix_table, row_id_column), (row_axis, row_axis_column), (column_axis, column_axis_column)
        ):
            if column not in table["columns"]:
                raise ValueError(f"Matrix ID column {column!r} is not selected for {table['table_id']!r}")

        def axis_keys(values, axis_name):
            normalized = [str(value) for value in values]
            if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
                raise ValueError(f"Matrix {axis_name} IDs must be non-empty and unique: {values}")
            return normalized

        row_ids = [record["values"][row_axis_column] for record in row_axis["records"]]
        column_ids = [record["values"][column_axis_column] for record in column_axis["records"]]
        matrix_row_ids = [record["values"][row_id_column] for record in matrix_table["records"]]
        matrix_columns = [column for column in matrix_table["columns"] if column != row_id_column]
        def axis_mapping(actual, expected):
            axis_keys(actual, "source")
            axis_keys(expected, "reference")
            if set(actual) == set(expected):
                return dict(zip(actual, actual)), "exact"
            def suffix_map(values):
                matches = [re.fullmatch(r"\D*(\d+)", value) for value in values]
                if not all(matches):
                    return None
                keys = [match.group(1) for match in matches]
                return dict(zip(keys, values)) if len(set(keys)) == len(keys) else None
            left, right = suffix_map(actual), suffix_map(expected)
            if left is not None and right is not None and set(left) == set(right):
                return {value: right[key] for key, value in left.items()}, "unique_complete_suffix"
            return None, "unresolved"
        row_map, row_basis = axis_mapping(matrix_row_ids, row_ids)
        column_map, column_basis = axis_mapping(matrix_columns, column_ids)
        relationship = dict(relationship)
        if row_basis == "unique_complete_suffix":
            relationship["row_id_mapping"] = row_map
        if column_basis == "unique_complete_suffix":
            relationship["column_id_mapping"] = column_map
        checks = {
            "matrix_table_id": matrix_id,
            "shape": [len(matrix_table["records"]), len(matrix_columns)],
            "expected_shape": [len(row_ids), len(column_ids)],
            "row_ids_aligned": row_map is not None, "column_ids_aligned": column_map is not None,
            "row_mapping_basis": row_basis, "column_mapping_basis": column_basis,
        }
        if checks["shape"] != checks["expected_shape"] or not checks["row_ids_aligned"] or not checks["column_ids_aligned"]:
            raise ValueError(f"Matrix validation failed: {checks}")
        validated_relationships.append(relationship)
        matrix_checks.append(checks)

    payload_tables.sort(key=lambda table: (table["file_index"], table["table_id"]))
    validation = {"status": "OK", "matrix_checks": matrix_checks}
    return {
        "route": route, "query": str(query or ""), "tables": payload_tables,
        "relationships": validated_relationships, "ignored_file_indices": sorted(ignored),
        "validation": validation,
    }


# CELL 11
def load_csv_documents(dataset_address):
    """Build one legacy document per source row in deterministic source order."""
    documents, source_order = [], 0
    for table in _load_tables(dataset_address):
        for source_row, row in table["frame"].iterrows():
            values = {str(column): str(row[column]) for column in row.index}
            documents.append(Document(
                page_content=json.dumps({"values": values}, ensure_ascii=False),
                metadata={"source": str(table["path"]), "source_order": source_order},
            ))
            source_order += 1
    return documents


def build_csvqa_components(
    dataset_address, system_prompt, tool_description, *, route, user_query="",
    result_sink=None,
):
    """Build the shared CSVQA tool in query-planned or legacy full-data mode."""
    route = normalize_route(route)
    planned = CSVQA_MODE_BY_ROUTE[route] == "planned"
    llm = make_llm()

    state = result_sink if result_sink is not None else {}

    def rag_answer(tool_query):
        documents = load_csv_documents(dataset_address)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "human",
                "Original user query (the only authority for subsets):\n{user_query}\n\n"
                "Agent retrieval request (must not narrow beyond the original query):\n{input}",
            ),
        ])
        result = create_stuff_documents_chain(llm, prompt).invoke({
            "input": tool_query, "user_query": user_query, "context": documents,
        })
        if not str(result or "").strip():
            raise RuntimeError("Legacy CSVQA model returned an empty result")
        return str(result)

    def qa_wrapper(tool_query):
        file_indices = None
        if str(tool_query).lstrip().startswith("{"):
            try:
                request = json.loads(tool_query)
            except json.JSONDecodeError as exc:
                raise ReActProtocolError("csvqa_action_input_format") from exc
            file_indices = request.get("file_indices")
        plan, planner_outputs, errors, status, fallback_reason = None, [], [], "LEGACY_FULL_DATA", None
        attempt_count = 0
        if planned:
            tables = _load_tables(dataset_address, file_indices=file_indices)
            attempt_count = 1
            raw_output = ""
            try:
                plan, raw_output = _ask_extraction_plan(
                    llm, route, user_query, tables, system_prompt
                )
                payload = _execute_plan(plan, tables, user_query, route)
                planner_outputs.append(raw_output)
                observation = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                status = "PLANNED"
            except (KeyError, TypeError, ValueError) as exc:
                planner_outputs.append(getattr(exc, "planner_output", raw_output))
                errors.append(str(exc))
                if CSVQA_MODE_BY_ROUTE[route] != "planned":
                    raise ValueError(
                        f"CSVQA extraction plan failed for {route}: {errors[-1]}"
                    ) from exc
                payload_tables = []
                for table in tables:
                    frame, columns = table["frame"], list(table["frame"].columns)
                    payload_tables.append({
                        "table_id": f"file_{table['file_index']}_view_0",
                        "file_index": table["file_index"], "file_name": table["path"].name,
                        "role": f"file_{table['file_index']}", "columns": columns,
                        "original_rows": len(frame), "returned_rows": len(frame),
                        "filters": {"logic": "and", "conditions": []},
                        "records": [{"source_row": int(index), "values": {
                            column: str(row[column]) for column in columns
                        }} for index, row in frame.iterrows()],
                    })
                fallback_reason = errors[-1] if errors else "planner did not produce a valid plan"
                observation = json.dumps({
                    "route": route, "query": str(user_query or ""), "tables": payload_tables,
                    "relationships": [], "ignored_file_indices": [],
                    "validation": {
                        "status": "FALLBACK_FULL_DATA", "planner_errors": errors,
                        "fallback_reason": fallback_reason,
                    },
                }, ensure_ascii=False, sort_keys=True)
                status = "FALLBACK_FULL_DATA"
        else:
            observation = rag_answer(tool_query)
        trace = {
            "version": "GPT4.1-LO", "mode": "planned" if planned else "legacy",
            "status": status, "requested_file_indices": file_indices, "plan": plan, "planner_attempt_count": attempt_count,
            "planner_outputs": planner_outputs, "planner_errors": errors,
            "validation_errors": errors, "fallback_reason": fallback_reason,
            "repair_count": 0, "retry_count": 0,
            "fallback_count": int(status == "FALLBACK_FULL_DATA"),
            "payload_hash": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
        }
        state.clear()
        state.update({"observation": observation, "trace": trace})
        return observation

    return llm, Tool(name="CSVQA", func=qa_wrapper, description=tool_description), state


# CELL 13
@lru_cache(maxsize=1)
def _get_classifier_retriever():
    documents = CSVLoader(
        file_path=str(PROJECT_ROOT / "Large_Scale_Or_Files/RAG_Examples_All.csv"),
        encoding="utf-8-sig",
        content_columns=["prompt", "Type"],
    ).load()
    if len(documents) < 5:
        raise ValueError("Classification reference requires at least five labeled examples")
    return FAISS.from_documents(documents, make_embeddings()).as_retriever(
        search_kwargs={"k": 5}
    )


def fileqa_retrieve(query):
    """Retrieve five raw classification examples without an additional LLM call.
    
    Args:
        query: Natural-language optimization problem or retrieval query.
    
    Returns:
        A formatted string containing the ranked retrieved examples.
    """
    retrieved = _get_classifier_retriever().invoke(str(query))
    if len(retrieved) != 5:
        raise RuntimeError(f"FileQA expected five examples, retrieved {len(retrieved)}")

    return "\n\n".join(
        f"[Retrieved example {rank}]\n{document.page_content}"
        for rank, document in enumerate(retrieved, start=1)
    )


few_shot_example = """
Example 1 — Network Revenue Management

Question:
What is the problem type of the following optimization problem? Text: There
are three products with corresponding revenues and independent demand
streams. The company decides how many requests to fulfil over a finite sales
horizon to maximize total revenue from fixed initial inventories.
Replenishment is not allowed, and fulfilment decisions are non-negative
integers.

Thought: I need to identify the mathematical structure. I should call FileQA
once to retrieve structurally similar labeled reference problems.

Action: FileQA

Action Input: finite-horizon revenue maximization with independent demand streams, fixed initial inventories, no replenishment, and integer demand-fulfilment decisions

Observation: The retrieved RAG_Examples_All examples contain finite-inventory
demand-fulfilment problems labeled Network Revenue Management.

Thought: Fixed inventories are allocated to demand fulfilment over a sales
horizon to maximize revenue. This is the defining NRM structure.

Final Answer: NRM


Example 2 — Mixture rather than Resource Allocation

Question:
What is the problem type of the following optimization problem? Text: A
manufacturer determines integer production quantities and integer batch
counts on several workstations. Each batch consumes machine capacity.
Production creates a byproduct that must either be sold or disposed of, and
the corresponding quantities must satisfy balance constraints.

Thought: This problem contains resource capacities, but I must check whether
it is a clean single-stage resource-allocation model or contains additional
interacting mechanisms. I should call FileQA once.

Action: FileQA

Action Input: integer production and batch-count decisions across multiple workstations with capacity constraints and coupled byproduct sale-or-disposal balance

Observation: The retrieved examples include resource-allocation and
non-standard production-planning models. The latter contain batch decisions,
multiple production components, and coupled balance constraints.

Thought: Ordinary capacity constraints alone could indicate RA. However,
integer batch counts, multiple workstations, and byproduct balance materially
change the formulation. Therefore, the problem is Mixture rather than RA.

Final Answer: Mixture


Example 3 — Others rather than Mixture

Question:
What is the problem type of the following optimization problem? Text: A
company selects a subset of independent projects. Each project has a profit
and consumes part of one available budget. Each project is represented by a
binary variable, and there are no precedence, dependency, assignment, batch,
flow, or multi-stage constraints.

Thought: I need to determine whether this is a hybrid model or a coherent
standard optimization family outside the five canonical classes. I should
call FileQA once.

Action: FileQA

Action Input: independent binary project selection maximizing profit under one budget constraint with no logical dependencies or additional coupling

Observation: The retrieved examples contain standard binary knapsack and
project-selection problems categorized as outside the five canonical
classes.

Thought: This is a clean knapsack-style model without interacting structures
or non-standard coupling. It should be Others, not Mixture.

Final Answer: Others
"""


prefix = f"""
You are the classification agent in LEAN-LLM-OPT. Classify the current
optimization problem into exactly one of the following seven semantic
classes:

- NRM: Network Revenue Management
- RA: Resource Allocation
- TP: Transportation Problem
- FLP: Facility Location Problem
- AP: Assignment Problem
- Mixture
- Others

You must follow the ReAct workflow and call FileQA exactly once.

FileQA retrieves structurally similar labeled optimization problems from
RAG_Examples_All. For the Action Input, provide a concise but complete structural
description that preserves the objective, decision variables, major
constraints, and important coupling mechanisms.

Retrieved examples are supporting evidence, not automatic votes. The current
problem description and its mathematical structure are authoritative.

Classify according to:

- the objective;
- the meanings and domains of the decision variables;
- the principal constraint families;
- coupling between decisions;
- whether the formulation contains multiple stages or additional modeling
  mechanisms.

Do not classify according to company names, application narratives,
filenames, CSV files, or isolated keywords.

Class definitions:

1. Network Revenue Management (NRM)

A problem that allocates fixed inventories or capacities to known or
stochastic demand streams in order to maximize revenue. Typical defining
features include demand fulfilment, finite inventory, a sales horizon, and
no replenishment.

2. Resource Allocation (RA)

A clean, primarily single-stage allocation of limited resources among
activities or products, usually with ordinary capacity constraints and an
additive objective.

Capacity constraints or integer production quantities alone do not make a
problem Mixture.

Do not choose RA when additional mechanisms materially change the
formulation, such as:

- batch-size or lot-count decisions;
- multiple interacting production stages, workstations, or devices;
- byproduct sale, disposal, or material-balance decisions;
- blending or proportion constraints interacting with production;
- logical dependencies or conditional decisions;
- selection followed by assignment;
- equilibrium-style balance equations;
- vehicle-count and shipment-quantity coupling.

3. Transportation Problem (TP)

A clean source-to-destination flow problem with source supplies, destination
demands, route shipment quantities, and route transportation costs.

Do not choose TP when the transportation structure is materially coupled
with mechanisms such as integer truck dispatch, vehicle-capacity decisions,
fixed-charge decisions, multi-stage transshipment, production decisions, or
other non-standard constraint families. Such a problem may be Mixture.

4. Facility Location Problem (FLP)

A problem containing facility-opening or location decisions with fixed
opening costs, together with assignment or shipment decisions that link
clients or demand points to opened facilities.

5. Assignment Problem (AP)

A clean one-to-one or matching-style assignment of agents, workers,
machines, products, or other entities to tasks.

If the problem first selects a subset of entities and then assigns them, or
combines assignment with multiple stages, logical dependencies, production,
flow, or other interacting constraint families, choose Mixture instead.

6. Mixture

A hybrid or non-standard optimization model in which multiple structures or
modeling mechanisms interact materially.

Choose Mixture when either:

(a) a canonical TP, NRM, RA, FLP, or AP structure is combined with an
important additional mechanism; or

(b) a non-core optimization model contains non-standard interacting
constraint families that prevent it from being a clean textbook instance.

Strong Mixture signals include:

- batch-size, lot-size, or integer-trip decisions coupled with quantities;
- logical implications, precedence, dependencies, or conditional decisions;
- subset selection coupled with assignment;
- blending or proportion requirements coupled with production or capacity;
- byproduct production, sale, disposal, or balance;
- multiple interacting workstations, devices, or production stages;
- equilibrium or cross-variable balance equations;
- truck-count or vehicle-capacity decisions coupled with transportation flow;
- several distinct decision families whose interactions materially affect
  the formulation.

A problem does not need to belong to two named canonical classes to be
Mixture. One recognizable base structure plus a material non-standard
coupling mechanism is sufficient.

7. Others

A clean and coherent optimization family outside NRM, RA, TP, FLP, and AP,
without the hybrid or non-standard interacting structure required for
Mixture.

Typical examples include:

- a standard diet problem;
- a pure knapsack or independent project-selection problem;
- a standard traveling salesperson problem;
- a standard workforce or shift-covering problem;
- a standard parallel-machine scheduling problem;
- another well-known outside family with one dominant mathematical
  structure.

A textbook blending model may be Others only when it is a standalone model
with ordinary ingredient-balance and quality constraints. If blending or
proportion decisions interact with production stages, capacities, equipment,
batch decisions, or other constraint families, choose Mixture.

Decision procedure:

1. Identify the objective, decision variables, and main constraints.

2. Determine whether the problem is a clean instance of NRM, RA, TP, FLP,
   or AP.

3. Even if a canonical base structure is present, check for a material
   additional mechanism such as batching, logical dependencies,
   selection-assignment coupling, byproducts, blending-production coupling,
   multiple stages, equilibrium equations, or vehicle-count coupling.

4. If such an additional mechanism materially changes the formulation,
   choose Mixture.

5. If the problem is a clean, well-known outside family with one dominant
   structure and no material hybrid coupling, choose Others.

6. When TP, NRM, RA, FLP, or AP cannot be confidently confirmed:
   - choose Mixture if multiple structures or non-standard constraints
     interact materially;
   - otherwise choose Others.

When deciding between Mixture and Others, choose Mixture if two different decision families or constraint mechanisms interact materially. Choose Others only when one standard textbook archetype explains all important variables and constraints.

7. Retrieved labels are evidence, not rules. Do not use majority voting.
   When retrieval conflicts with the current mathematical formulation,
   prioritize the current formulation.

Important boundary rules:

- Multiple ordinary resource-capacity constraints alone are not sufficient
  for Mixture; a clean allocation model remains RA.
- Integer variables alone are not sufficient for Mixture.
- A canonical structure plus a material additional decision mechanism is
  sufficient for Mixture.
- Mixture and Others are separate semantic labels even if they currently
  share the same downstream workflow route.

Return exactly one of the following label tokens in the Final Answer:

NRM, RA, TP, FLP, AP, Mixture, Others

Do not include explanations, punctuation, or multiple candidate labels in
the Final Answer.

Examples of the required ReAct process:

{few_shot_example}
"""


suffix = """
Begin!

Question: {input}
{agent_scratchpad}
"""


@lru_cache(maxsize=1)
def _get_classification_agent():
    llm = make_llm()
    tool = Tool(
        name="FileQA",
        func=fileqa_retrieve,
        description=(
            "Return five structurally similar RAG examples and their labels. "
            "Use the complete optimization problem text as input."
        ),
    )
    return initialize_agent(
        tools=[tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": prefix, "suffix": suffix}, verbose=True,
        handle_parsing_errors=False, max_iterations=4, return_intermediate_steps=True,
    )


def normalize_problem_class(value):
    """Normalize a raw label to one of the seven benchmark classes.
    
    Args:
        value: Input value to normalize, parse, or inspect.
    
    Returns:
        The canonical class label.
    """
    candidate = str(value or "").strip()
    final = re.findall(r"(?:Final Answer|Answer|Label|Class)\s*:\s*([^\n]+)", candidate, re.I)
    if final:
        candidate = final[-1].strip()
    upper = re.sub(r"\s+", " ", candidate.strip("`*_ .,:;").upper())
    aliases = {
        "TP": "TP", "TRANSPORTATION": "TP", "TRANSPORTATION PROBLEM": "TP",
        "NRM": "NRM", "NETWORK REVENUE MANAGEMENT": "NRM",
        "NETWORK REVENUE MANAGEMENT PROBLEM": "NRM",
        "RA": "RA", "RESOURCE ALLOCATION": "RA", "RESOURCE ALLOCATION PROBLEM": "RA",
        "FLP": "FLP", "UFLP": "FLP", "FACILITY LOCATION": "FLP",
        "FACILITY LOCATION PROBLEM": "FLP",
        "UNCAPACITATED FACILITY LOCATION PROBLEM": "FLP",
        "UNCAPACITED FACILITY LOCATION PROBLEM": "FLP",
        "AP": "AP", "ASSIGNMENT": "AP", "ASSIGNMENT PROBLEM": "AP",
        "MIXTURE": "Mixture", "OTHERS": "Others",    }
    if upper in aliases:
        return aliases[upper]
    if upper.startswith("OTHERS") or upper.startswith("SALES-BASED LINEAR PROGRAMMING"):
        return "Others"
    matches = [
        label for label, pattern in [
            ("Mixture", r"\bMIXTURE\b"),
            ("NRM", r"\bNRM\b|NETWORK REVENUE MANAGEMENT"),
            ("RA", r"\bRA\b|RESOURCE ALLOCATION"),
            ("TP", r"\bTP\b|TRANSPORTATION"),
            ("FLP", r"\bFLP\b|\bUFLP\b|FACILITY LOCATION"),
            ("AP", r"\bAP\b|ASSIGNMENT"),
            ("Others", r"\bOTHERS"),
        ]
        if re.search(pattern, upper)
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Cannot unambiguously normalize class: {value!r}")


def class_to_workflow_route(label):
    """Map a semantic class to an executable workflow route.
    
    Args:
        label: Semantic problem-class label.
    
    Returns:
        The route name; Mixture and Others both map to Others.
    """
    normalized = normalize_problem_class(label)
    return "Others" if normalized in {"Mixture", "Others"} else normalized


def invoke_classifier(query):
    """Run the ReAct classification agent and validate its tool-use contract.
    
    Args:
        query: Natural-language optimization problem or retrieval query.
    
    Returns:
        A dictionary containing the raw output, normalized label, FileQA call count, and trace.
    
    Raises:
        RuntimeError: If the classifier violates the one-FileQA-call contract or emits an invalid label.
    """
    response = invoke_react_protocol(_get_classification_agent, str(query), "classification")
    steps = response.get("intermediate_steps", []) if isinstance(response, dict) else []
    fileqa_calls = sum(
        getattr(step[0], "tool", None) == "FileQA"
        for step in steps if isinstance(step, tuple) and step
    )
    if fileqa_calls != 1:
        raise RuntimeError(f"Classifier must call FileQA exactly once; observed {fileqa_calls}")
    output = str(response.get("output", "") if isinstance(response, dict) else response).strip()
    return {
        "output": output,
        "normalized_label": normalize_problem_class(output),
        "fileqa_calls": fileqa_calls,
        "intermediate_steps": steps,
    }


# CELL 15
def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Original ReAct with mandatory current CSVQA and user-authorized protocol restarts."""
    route = normalize_route(route)
    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    tool_history, current_observations = [], []
    original_tool_func = qa_tool.func
    def invoke_csvqa(tool_query):
        entry = {"request": str(tool_query)}
        tool_history.append(entry)
        observation = original_tool_func(tool_query)
        entry.update(trace=dict(csvqa_result.get("trace", {})), observation=observation)
        if CSVQA_MODE_BY_ROUTE[route] == "planned":
            current_observations.append(json.loads(observation))
        return observation
    qa_tool.func = invoke_csvqa
    sources = dict(enumerate(raw.strip() for raw in str(dataset_address).splitlines() if raw.strip()))
    prefix += ("\nFor the CURRENT problem, call CSVQA at least once before the Final Answer. "
               "Demonstration Observations are historical and cannot supply current data. "
               "You may call CSVQA multiple times for different current CSV files. "
               "Use the original user description as Action Input to read all CSV files, or use "
               'JSON Action Input with keys "query" and "file_indices" to read selected zero-based indices. '
               "Use a concise symbolic model and exact Data Mapping with table_id and column names. "
               "Bind every parameter to source data or a query-defined expression; do not invent a missing limit. "
               "If total capacity denotes listed per-entity capacities, retain their bounds and their sum. "
               "Preserve all variable domains, constraints, objective sense and additive constants. "
               "Define each index set from ALL current returned entities unless the original query restricts it. "
               "Example counts and profile samples must not determine the current entity set. "
               "Use optional matrix alias mappings only when supplied; otherwise preserve exact raw IDs.\n"
               "Current source index-to-path mapping: " + escape_braces(json.dumps(sources)))
    prefix += '\nApply unconditional quantity/resource bounds unconditionally. Only use activation-conditioned bounds when the current query explicitly makes them conditional; an activation fee alone does not make an otherwise unconditional bound conditional.'
    suffix = """Begin!

CURRENT USER DESCRIPTION START
{input}
CURRENT USER DESCRIPTION END

CURRENT CSVQA DATA STATE: {current_data_state}

If the state is NOT_LOADED, a Final Answer is forbidden. Respond with:
Thought: I need the current source data.
Action: CSVQA
Action Input: <copy only the user description between START and END, or provide the specified JSON>
Do not copy markers, protocol instructions, examples or status into Action Input.
When the state is READY, request another CSV if needed, or respond with this envelope:
Thought: I have the required current data and can formulate the model.
Final Answer:
<the complete symbolic mathematical model and Data Mapping>
The literal label "Final Answer:" is mandatory before every mathematical-model heading.
Never output a bare model or markdown heading before these labels.
Keep numerical data in the Observation; do not repeat its tables or enumerate parameters.
Do not treat any demonstration's Observation as current problem data.
{agent_scratchpad}"""
    def create_agent():
        csvqa_result.clear()
        current_observations.clear()
        retry_note = ("\nA previous protocol attempt was discarded. Follow the exact Thought/Action/Action Input "
                      "or Thought/Final Answer envelope and call CSVQA before your final model."
                      if REACT_PROTOCOL_EVENTS else "")
        agent = initialize_agent(
            tools=[qa_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={"prefix": prefix + retry_note, "suffix": suffix}, verbose=True,
            handle_parsing_errors=False, return_intermediate_steps=True, early_stopping_method="force",
        )
        agent.agent.llm_chain.prompt = agent.agent.llm_chain.prompt.partial(
            current_data_state=lambda: "READY" if csvqa_result.get("observation") else "NOT_LOADED",
        )
        return agent
    try:
        result = invoke_react_with_required_csvqa(create_agent, query, route)
        if len(current_observations) > 1:
            tables = {t["table_id"]: t for payload in current_observations for t in payload["tables"]}
            relationships = {r["matrix_table_id"]: r for payload in current_observations
                             for r in payload.get("relationships", [])}
            payload = {**current_observations[-1], "tables": list(tables.values()),
                       "relationships": list(relationships.values()), "ignored_file_indices": []}
            csvqa_result["observation"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        evidence = dict(csvqa_result)
        evidence["trace"] = {**evidence.get("trace", {}), "formulation_protocol": "ReAct",
                             "csvqa_call_count": len(tool_history), "tool_calls": tool_history,
                             **react_protocol_record_fields()}
        exc.csvqa_result = evidence
        if not evidence.get("observation"):
            exc.pipeline_stage = "data_extraction"
        raise
    trace = csvqa_result.setdefault("trace", {})
    trace.update(formulation_protocol="ReAct", csvqa_call_count=len(tool_history), tool_calls=tool_history,
                 fallback_count=sum(c.get("trace", {}).get("fallback_count", 0) for c in tool_history),
                 **react_protocol_record_fields())
    trace["payload_hash"] = hashlib.sha256(csvqa_result["observation"].encode("utf-8")).hexdigest()
    return {"formulation": result["output"], **csvqa_result}


@lru_cache(maxsize=1)
def _load_rag_table():
    """Read the unified examples once, preserving every field as text."""
    frame = read_csv_compat(RAG_EXAMPLES_ALL_PATH, dtype=str, keep_default_na=False)
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RAG example CSV is missing columns: {sorted(missing)}")
    frame["Type"] = frame["Type"].map(normalize_route)
    return frame


def load_rag_examples(route):
    """Return one canonical route; UFLP maps to FLP and Mixture to Others."""
    frame = _load_rag_table()
    examples = frame.loc[frame["Type"].eq(normalize_route(route))].copy()
    if examples.empty:
        raise ValueError(f"No RAG examples found for route {route!r}")
    return examples


@lru_cache(maxsize=None)
def _get_rag_store(route):
    """Share a semantic index across modeling and code generation for each route."""
    documents = [
        Document(
            page_content=example["prompt"],
            metadata={"example_json": json.dumps(example, ensure_ascii=False)},
        )
        for example in load_rag_examples(route).to_dict("records")
    ]
    # Embed the problem only: code/formulation length must not dominate similarity.
    return FAISS.from_documents(documents, make_embeddings())


def retrieve_rag_examples(route, query, k=1):
    """Retrieve structured example dictionaries, never parse CSVLoader display text."""
    if k < 1:
        return []
    route = normalize_route(route)
    documents = _get_rag_store(route).similarity_search(
        str(query), k=min(k, len(load_rag_examples(route))),
    )
    return [json.loads(document.metadata["example_json"]) for document in documents]


def retrieve_similar_texts(query, retriever):
    # The separate query-only reference dataset still uses its original retriever.
    return [doc.page_content for doc in retriever.invoke(query)]


def escape_braces(text):
    """Protect literal example braces when embedding text into an agent template."""
    return text.replace("{", "{{").replace("}", "}}")


def rag_example_observation(example):
    """Prefer the curated evidence, including its original source-row identifiers."""
    evidence = example["Required Data"].strip()
    if evidence:
        return evidence
    blocks = []
    for raw in example["Data_address"].splitlines():
        if raw.strip():
            frame = read_csv_compat(raw.strip(), dtype=str, keep_default_na=False)
            blocks.append(f"Source: {raw.strip()}\n{frame.to_json(orient='records', force_ascii=False)}")
    return "\n\n".join(blocks)


def build_formulation_examples(route, query, k=1):
    """Render the common ReAct example format once for all CSV modeling routes."""
    examples = []
    for example in retrieve_rag_examples(route, query, k):
        block = f"""Question: Based on the following problem description and data, formulate a complete mathematical model. {example['prompt']}

Thought: Retrieve all data needed by the question, in source order. Preserve identifiers, coefficients, and any additional constraints. If the question does not request a subset, retrieve the whole dataset.

Action: CSVQA

Action Input: Retrieve all data {example['Related'].strip()} required to formulate this model. Preserve source order and exact identifiers and values, without simplification or abbreviation.

Observation: {rag_example_observation(example)}

Thought: Use the retrieved data as parameters and include all query-specific constraints. Choose variable domains from the question. Return the complete mathematical model in markdown, with $ or $$ for mathematical expressions and without extra explanations or LaTeX environments.

Final Answer:
{example['Label']}"""
        # Prefix is itself a PromptTemplate; escape the entire block exactly once.
        examples.append(escape_braces(block))
    return "\n\n".join(examples)


def get_NRM_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("NRM", query, k=1)
    csvqa_system_prompt = CSVQA_PLANNED_PROMPTS["NRM"]
    csvqa_tool_description = CSVQA_TOOL_DESCRIPTIONS["NRM"]

    route_instruction = """Call CSVQA exactly once and return an ABSTRACT model. Define
index sets, parameters, variables, objective, and constraints symbolically. Include a short Data
Mapping that identifies every source by exact table_id and exact column name. Do not copy record
values or literal record counts."""
    prefix = f"""You are an assistant that formulates mathematical optimization models.

    The examples below illustrate model structure only; never copy their data.

    {few_shot_examples}

    {route_instruction}
    Use the user query as the authority for objective sense, variable domains, boundary semantics,
    and additional constraints. Do not invent missing data or silently omit returned data.
    """

    suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "NRM", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )


# CELL 17
def get_RA_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("RA", query, k=3)
    csvqa_system_prompt = CSVQA_PLANNED_PROMPTS["RA"]
    csvqa_tool_description = CSVQA_TOOL_DESCRIPTIONS["RA"]

    route_instruction = """You MUST call CSVQA exactly once before the Final Answer and use all returned rows.
Keep the supplied file order and original row order. Never sort or reset an index.
Use an explicit business ID column when present; never synthesize business IDs from row numbers.
Do not infer a file's semantic role only from its filename or position.
Align profits, resource consumption, and capacities through explicit column/ID keys.
Every resource-capacity dimension must receive a capacity constraint, and each
activity must appear in the objective and all applicable resource constraints.
Infer integer/binary/continuous domains from the user description and data.
An item count or number of units to order, produce, allocate, or develop is a nonnegative integer
unless the original query explicitly permits fractional quantities or describes divisible material.
The words scale, quantity or allocation alone do not establish continuous variables.
Distinguish a global activity decision from separate resource-by-activity allocations. When independent
warehouses, machines or other resources can each allocate to activities, use decisions indexed by
both resource and activity; do not constrain one identical global expression by every resource's limit.
Each resource-indexed capacity constraint must depend on that resource through its decisions or
consumption coefficients. For different resource dimensions consumed by one global activity, retain
global activity variables and the dimension-specific consumption coefficients instead.
Never invent a total capacity, default RHS, or omitted consumption coefficient.
Return a concise ABSTRACT mathematical model with symbolic index sets, parameters, variables,
objective and constraints. Include a Data Mapping for every parameter using exact table_id and
column names from CSVQA_DATA. Keep every resource dimension and original business identifier.
Do not enumerate record values or hard-code entity counts; Python supplies the source records.
Ignore unrelated administrative columns; numeric columns are not automatically resources or costs.
"""
    prefix = f"""You are an assistant that formulates mathematical optimization models.

    The examples below illustrate model structure only; never copy their data.

    {few_shot_examples}

    {route_instruction}
    Use the user query as the authority for objective sense, variable domains, boundary semantics,
    and additional constraints. Do not invent missing data or silently omit returned data.
    """

    suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "RA", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )


# CELL 19
def get_TP_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("TP", query, k=3)
    csvqa_system_prompt = CSVQA_LEGACY_SYSTEM_PROMPT
    csvqa_tool_description = CSVQA_LEGACY_TOOL_DESCRIPTION

    route_instruction = """Call CSVQA at least once and return a complete numerical formulation. Retrieved
Information must contain every identifier and coefficient required by downstream code. Preserve
source order and do not invent, aggregate, or silently omit data."""
    prefix = f"""You are an assistant that formulates mathematical optimization models.

    The examples below illustrate model structure only; never copy their data.

    {few_shot_examples}

    {route_instruction}
    Use the user query as the authority for objective sense, variable domains, boundary semantics,
    and additional constraints. Do not invent missing data or silently omit returned data.
    """

    suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "TP", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )


# CELL 21
def get_AP_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("AP", query, k=1)
    csvqa_system_prompt = 'Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead.Context: {context}'
    csvqa_tool_description = 'Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query.'

    prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {few_shot_examples}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

            When you need to retrieve information from the CSV file, use the provided tool.

            """

    suffix = """

            Begin!

            User Description: {input}
            {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "AP", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )


# CELL 23
def get_FLP_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("FLP", query, k=1)
    csvqa_system_prompt = ('Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved.Context: {context}' + ' Preserve every facility ID, customer ID, FixedCost, Capacity, Demand, and cost-matrix axis together with its source-row position. FixedCost and Capacity must use the same facility ID; the matrix row and column IDs must explicitly match facilities and customers, with its source orientation and shape retained. Model a two-dimensional shipment decision only. Use a stated capacity when present; an absent capacity is unresolved evidence, not zero. Do not transpose, truncate, pad, zero-fill, or infer an extra commodity/product axis.')
    csvqa_tool_description = 'Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query.'

    prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

            Please refer to the following example and generate the answer in the same format:

            {few_shot_examples}

            Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

            When you need to retrieve information from the CSV file, use the provided tool.

            """

    suffix = """
            Begin!

            User Description: {input}
            {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "FLP", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )


# CELL 25
def build_few_shot_Other(examples, t="Model"):
    """Render already-structured examples; inserted prompt values need no brace escaping."""
    field = "Label" if t == "Model" else "Code"
    thought = (
        "I need to create an Abstract Model Plan based on the query and CSV schema."
        if t == "Model" else
        "I need to generate a single, complete, executable Gurobi Python code block."
    )
    return "\n\n".join(
        f"<EXAMPLE>\nQuestion: {example['prompt']}\nThought: {thought}\n"
        f"Final Answer:\n{example[field]}\n</EXAMPLE>"
        for example in examples if example[field].strip()
    )


def csv_schema_preview(dataset_address, query=""):
    """Preview plus full-file evidence; matching never chooses or filters the model data."""
    normalize = lambda value: re.sub(r"\s+", " ", str(value)).strip().casefold()
    normalized_query = normalize(query)
    quoted = re.findall(r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\x27([^\x27\n]+)\x27(?!\w)', query)
    names = {normalize(next(part for part in group if part)) for group in quoted}
    blocks = []
    for raw in normalize_data_address(dataset_address).splitlines():
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
        typed = read_csv_compat(path)
        frame = read_csv_compat(path, dtype=str, keep_default_na=False)
        stats, columns, terms = {}, {}, set(names)
        for column in frame:
            values = frame[column]
            normalized = values.map(normalize)
            columns[column] = normalized
            present = values[normalized.ne("")]
            numbers = pd.to_numeric(present, errors="coerce")
            stats[column] = {"missing": int(normalized.eq("").sum()), "unique_nonempty": int(present.nunique())}
            if len(present) and numbers.notna().all():
                stats[column]["numeric_range"] = [float(numbers.min()), float(numbers.max())]
            # Unquoted evidence must literally name a real column or complete nonnumeric value.
            candidates = [column] + present.unique().tolist()
            for value in candidates:
                term = normalize(value)
                if term and not re.fullmatch(r"[-+\d.,%]+", term) and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized_query):
                    terms.add(term)
        evidence = []
        for term in sorted(terms):
            matches = []
            for column, values in columns.items():
                exact, prefix, contains = values.eq(term), values.str.startswith(term), values.str.contains(term, regex=False)
                if contains.any():
                    matches.append({"column": column, "exact": int(exact.sum()), "prefix": int(prefix.sum()),
                                    "contains": int(contains.sum()), "examples": frame.loc[contains, column].drop_duplicates().head(3).tolist()})
            evidence.append({"term": term, "matching_columns": matches,
                             "exact_matching_columns": sum(item["exact"] > 0 for item in matches)})
        blocks.append(
            f"File: {path.resolve()}\nCSV delimiter: comma; first line consumed as header.\nTotal rows: {len(frame)}\nColumns: {list(frame.columns)}\n"
            f"Parsed column types: {typed.dtypes.astype(str).to_dict()}\n"
            f"Preview only (first 10 rows):\n{frame.head(10).to_string(index=False)}\n"
            f"Full-file column statistics: {json.dumps(stats, ensure_ascii=False)}\n"
            "Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.\n"
            "Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.\n"
            f"Query-name evidence: {json.dumps(evidence, ensure_ascii=False)}"
        )
    if not blocks:
        raise ValueError("No external CSV paths were supplied")
    return "\n\n---\n\n".join(blocks)


def get_Others_response(user_query: str, dataset_address: str):
    examples = retrieve_rag_examples("Others", user_query, k=3)
    llm2 = make_llm()

    try:
        print("\n[Gurobi Pipeline Step 1/3]: Getting CSV Scheme...")

        schema = csv_schema_preview(dataset_address, query=user_query)
        if re.search(r"\bsymmetric\b", user_query, re.I):
            schema += ("\nWhen the query explicitly states symmetry, align matrix axes by entity ID. "
                       "Fill a missing off-diagonal entry from its present transpose; preserve zeros. "
                       "Reject pairs with both entries missing or conflicting values beyond numeric tolerance. "
                       "Without explicit symmetry, preserve direction and never mirror or zero-fill entries.")
        if re.search(r"\bshut(?:down| down)\b", user_query, re.I):
            schema += "\nMinimum-down-time restrictions activate only on an on-to-off transition; down-time constraints alone must never forbid an always-on sequence."
        if "\\sum" in user_query:
            schema += "\nPreserve the query's explicit summation scope: aggregate every summed index inside the expression; never replace a summed index with a separate family of constraints."
        if re.search(r"\bworkers?\b", user_query, re.I) and re.search(r"\btasks?\b", user_query, re.I):
            schema += "\nIn a worker-task matrix, distinguish embedded row/column-axis captions from actual workers. Select workers by their identifiers, exclude label rows, then validate worker counts; never truncate by position."
        print("\n[Gurobi Pipeline Step 2/3]: Constructing Abstract Model...")

        few_shot_block_abstract = build_few_shot_Other(examples, t='Model')
        print(f"[Few-Shot Abstract Examples]:\n{few_shot_block_abstract}...")
        
        abstract_model_template = """
You are an expert optimization modeler.
Your task is to create an "Abstract Model Plan" based on the user's query and the CSV Schema (data structure).
This plan is *not* Gurobi code or mathematical formulas, but a clear, step-by-step reasoning process in English.

[Examples]
{few_shot_examples}

[Current Task]
User Query: {query}

CSV Schema:
{schema}

An explicitly enumerated complete entity set in the query is authoritative: extra CSV entities
must not enlarge that set unless the query explicitly delegates its definition to the data.
For a one-off finite-horizon schedule, change/ramp constraints compare consecutive modeled periods. An initial
binary on/off state is not a pre-horizon continuous quantity: do not invent an initial continuous
level or a first-period ramp limit unless the query explicitly supplies that continuous boundary.
For recurring daily/cyclic operation, preserve wrap-around coverage across the last/first period
when a working shift spans the boundary; do not impose a fictitious empty start to a recurring day.
State and implement the original objective sense and complete value expression, including constants.
Do not offer sign-reversed or constant-shifted optimizer-equivalent objectives as alternatives.
Examples illustrate style; the current query and schema determine the actual scope, indices and boundaries.

Data selection: state for each entity table whether all rows or a subset is required; name the
filter column, exact condition, entity key, and joins. Preserve shared resource/parameter tables.
Without a query-requested filter, use the complete entity set. A preview is never the selected set.
Matching statistics are evidence, not instructions: do not broaden a zero-match condition to
substring matching or all rows. Check selected keys and applicable counts at runtime; recompute
combined conditions rather than treating individual counts as intersection counts.

[Your Output]
You must strictly follow this format for your "Abstract Model Plan" output:

[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to {{{{Your analysis}}}}.
2.  **Identify Model Type:** Based on the query, this is a {{{{e.g., LP, MIP, Fixed-Charge, Blending}}}} problem.
3.  **Define Index Sets:** The primary indices are {{{{e.g., Products, Workers, Sources, Destinations}}}}.
4.  **Define Decision Variables:**
    -   `x[i]` = {{{{Describe first variable, e.g., 'quantity of product i'}}}}. Type: {{{{GRB.CONTINUOUS / GRB.INTEGER}}}}.
    -   `y[i]` = {{{{Describe second variable, e.g., 'if product i is produced'}}}}. Type: {{{{GRB.BINARY}}}}.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (e.g., profit) will come from column(s): {{{{e.g., 'Price', 'Production Cost'}}}}.
    -   Constraint coefficients (e.g., resource use) will come from: {{{{e.g., 'Resource 1', 'Available Time'}}}}.
    -   Constraint RHS (limits) will come from: {{{{e.g., 'Max Demand'}}}}.
6.  **Formulate Objective:** {{{{Describe objective, e.g., 'Maximize sum((schema['Price'] - schema['Cost']) * x[i] - schema['FixedCost'] * y[i])'}}}}.
7.  **Formulate Constraints:**
    -   Constraint 1 (e.g., Resource Limit): {{{{Describe constraint 1, e.g., 'sum(schema['Resource 1'][i] * x[i]) <= schema['Available']'}}}}.
    -   Constraint 2 (e.g., Linking): {{{{Describe constraint 2, e.g., 'x[i] <= M * y[i])'}}}}.
    -   ... (Other constraints) ...
[Abstract Model Plan END]
"""
        abstract_model_template += '\nApply unconditional quantity/resource bounds unconditionally. Only use activation-conditioned bounds when the current query explicitly makes them conditional; an activation fee alone does not make an otherwise unconditional bound conditional.'

        abstract_prompt = PromptTemplate(
            template=abstract_model_template,
            input_variables=["query", "schema", "few_shot_examples"]
        )
        
        abstract_model_chain = LLMChain(llm=llm2, prompt=abstract_prompt)
        
        abstract_model_plan = abstract_model_chain.run(
            query=user_query,
            schema=schema,
            few_shot_examples=few_shot_block_abstract
        )
        print(f"[Observation]:\n{abstract_model_plan}")
        
        print("\n[Gurobi Pipeline Step 3/3]: Generating Gurobi Code...")
        
        few_shot_block_code = build_few_shot_Other(examples, t='Code')
        
        code_gen_template = """
You are an expert Gurobi programmer.
Your task is to strictly follow the User Query, CSV Schema, and "Abstract Model Plan" to translate them into a single, complete, executable Gurobi Python code block.
The code must start with ```python and end with ```.
The code must include the necessary imports from gurobipy, pandas, numpy, re, math or json. Do not import sys or os.
Create exactly one Gurobi model in m or model and call its optimize() exactly once, both at top level. Do not wrap model creation or solving in functions or try/except; let errors propagate to the execution harness. Do not write files or call computeIIS().
The code must read all supplied CSV files needed by the model, at these exact newline-separated paths:
{dataset_address}
Read their complete contents at runtime. The schema preview is not the full dataset.
Preserve explicit business identifiers and matrix axes; do not invent filenames or columns.
Data-reading contract: these files were parsed with pandas.read_csv(path, sep=',').
The displayed preview is space-aligned for readability; it does not imply a whitespace delimiter.
pandas already consumes the CSV header. Select entity and parameter rows through identifier values
or validated predicates, never guessed iloc offsets or a guessed number of header rows.
Normalize identifier types explicitly on both sides of every lookup/join, including matrix row
indices versus column labels. Preserve the actual identifiers and derive dimensions from all rows.
Series.apply receives one cell value, not a row; use vectorized column predicates or DataFrame.apply
with axis=1 when a predicate needs a row. Every comprehension must use the indices it binds, not
stale variables from a preceding loop. Do not silently skip failed conversions or missing coefficients.
Use setObjective for the exact metric requested by the query, including all additive constants,
signs and units. An optimizer-equivalent objective is not an equivalent reported objective value;
do not drop constant terms and then merely restore them in printed output.
The code must *fully* implement all variables, objectives, and constraints from the "Abstract Model Plan".

[Examples of the Full Process]
{few_shot_examples}

[User Query]
{query}

[CSV Schema]
{schema}

[Abstract Model Plan]
{abstract_plan}

An explicitly enumerated complete entity set in the query is authoritative: extra CSV entities
must not enlarge that set unless the query explicitly delegates its definition to the data.
For a one-off finite-horizon schedule, change/ramp constraints compare consecutive modeled periods. An initial
binary on/off state is not a pre-horizon continuous quantity: do not invent an initial continuous
level or a first-period ramp limit unless the query explicitly supplies that continuous boundary.
For recurring daily/cyclic operation, preserve wrap-around coverage across the last/first period
when a working shift spans the boundary; do not impose a fictitious empty start to a recurring day.
State and implement the original objective sense and complete value expression, including constants.
Do not offer sign-reversed or constant-shifted optimizer-equivalent objectives as alternatives.
Examples illustrate style; the current query and schema determine the actual scope, indices and boundaries.

Data selection: state for each entity table whether all rows or a subset is required; name the
filter column, exact condition, entity key, and joins. Preserve shared resource/parameter tables.
Without a query-requested filter, use the complete entity set. A preview is never the selected set.
Matching statistics are evidence, not instructions: do not broaden a zero-match condition to
substring matching or all rows. Check selected keys and applicable counts at runtime; recompute
combined conditions rather than treating individual counts as intersection counts.

Reserve m and model for the Gurobi model; never reuse them as loop variables or other values.
Keep business identifiers as Python dictionary keys, separate from Gurobi names.
Use name='' for addVars/addConstrs so business keys are not expanded into solver names;
use only short ASCII names for individually named variables and constraints.
Never invent, truncate, or replace required coefficients with dummy or random data.
Before optimization, validate coefficient dimensions and identifier coverage against decision index sets;
raise an explicit error for missing data instead of guessing, padding, or silently dropping entries.

[Your Gurobi Code]
"""

        code_gen_template += "\n" + CSV_SOLVER_INSTRUCTIONS + MODEL_INTERFACE_INSTRUCTIONS

        code_gen_prompt = PromptTemplate(
            template=code_gen_template,
            input_variables=["query", "schema", "abstract_plan", "dataset_address", "few_shot_examples"]
        )
        
        code_gen_chain = LLMChain(llm=llm2, prompt=code_gen_prompt)
        
        final_code = code_gen_chain.run(
            query=user_query,
            schema=schema,
            abstract_plan=abstract_model_plan,
            dataset_address=dataset_address,
            few_shot_examples=few_shot_block_code 
        )
        
        if "```python" in final_code:
            code_block = final_code.split("```python", 1)[1]
            if "```" in code_block:
                code_block = code_block.split("```", 1)[0]
            final_answer = "```python\n" + code_block.strip() + "\n```"
        else:
            if not final_code.strip().startswith("import"):
                print(f"[Warning] Step 3 output was not a valid code block. Output: {final_code[:200]}...")
                final_answer = f"Error: Code generation failed. LLM returned non-code output:\n{final_code}"
            else:
                print("[Warning] Step 3 output missed ```python tag, adding it.")
                final_answer = "```python\n" + final_code.strip() + "\n```"
            
        print(f"[Final Answer]:\n{final_answer}")
        return {
            "formulation": abstract_model_plan, "code": final_answer, "observation": schema,
            "trace": {"status": "LEGACY_SCHEMA", "planner_attempt_count": 0,
                      "payload_hash": hashlib.sha256(schema.encode()).hexdigest()},
        }

    except Exception as e:
        print(f"[Gurobi Pipeline Error]: {e}")
        raise


def get_others_without_CSV_response(query):
    llm = make_llm()

    loader = CSVLoader(file_path=str(PROJECT_ROOT / "Large_Scale_Or_Files/RAG_Example_Others_Without_CSV.csv"), encoding="utf-8")
    documents = filter_loto_query_only_documents(loader.load())

    embeddings = make_embeddings()
    vectors = FAISS.from_documents(documents, embeddings)

    retriever = vectors.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    qa_tool = Tool(
        name="ORLM_QA",
        func=qa_chain.invoke,
        description=(
            "Use this tool to answer Querys."
            "Provide the Query as input, and the tool will retrieve the relevant information from the file and use it to answer the Query."
        ),
    )

    few_shot_examples = []
    similar_results = retrieve_similar_texts(query, retriever)

    for content in similar_results:

        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip()
        split_at_address = split_at_formulation[1].split("Label:", 1)
        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip() 

        label = label.replace("{", "{{").replace("}", "}}")
        example = (
            "<EXAMPLE>\n"
            f"Question: {problem_description}\n\n"
            "Thought: Read the question and 1) identify the goal (minimize time/cost/crew or maximize throughput/value) and collect per-unit coefficients; 2) define decision variables and pick domains: counts of trips/units/vehicles are nonnegative integers, yes or no choices are binary, divisible flows or weights are nonnegative reals; 3) write the linear objective from the coefficients; 4) add constraints in this order: demand or target (exactly, at least, at most), capacity or supply upper bounds, flow conservation or stage linking across nodes or arcs, and any share or ratio limits rewritten as linear inequalities using the given per-unit rates, plus any minimum or maximum usage; 5) add nonnegativity and the chosen integrality or binary domains; 6) output only the LP: objective first, then constraints line by line with brief labels if needed, then a final line stating the variable domains."
            "Final Answer:\n"
            f"{label}\n"
            "</EXAMPLE>"
        )

        example = example.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append(example)

    prefix = (
    f"{few_shot_examples}\n\n"
    """ 
    Use the following triggers to identify problem structures and apply the corresponding mathematical formulations.
    (1) INTEGER trigger:
    Question: A factory can run two machine types. How many of each machine should be installed given budget and space limits? Maximize output.
    Final Answer: 
    $\\max\\; p_1 x_1 + p_2 x_2$
    $\\text{{s.t. }} a_1 x_1 + a_2 x_2 \\le B,\\; s_1 x_1 + s_2 x_2 \\le S$
    $x_1, x_2 \\in \\mathbb{{Z}}_+.$
    (2) MULTI-PERIOD FLOW trigger:
    Question: Multi-period production with inventory and backorders. Costs for production/holding/backorder. Initial and terminal conditions given.
    Final Answer:
    \textbf{{Indices: }} t\\in T=\\{{1,\\dots,n\\}}. \\
    \textbf{{Given: }} d_t,\\ I_0,\\ B_0,\\ \\dots \\
    \textbf{{Vars: }} x_t\\ge0,\\ I_t\\ge0,\\ B_t\\ge0. \\
    \\min \\sum_t (c x_t + h I_t + p B_t) \\
    \text{{s.t. }} I_t - B_t = I_{{t-1}} - B_{{t-1}} + x_t - d_t,\\ \forall t \\
    I_n \\ge I^{{\\min}},\\ B_n=0.
    (3) LOGIC+BINARY trigger:
    Question: Choose exactly one option from set P and at least K items from set V, with quantities and budget.
    Final Answer:
    \textbf{{Sets: }} i\\in P,\\ j\\in V. \\ \textbf{{Vars: }} q_i,q_j\\ge0;\\ y_i\\in\\{{0,1\\}}, z_j\\in\\{{0,1\\}}.\\
    \\max \\sum_j f_j q_j \\
    \text{{s.t. }} \\sum_i y_i = 1,\\ \\sum_j z_j \\ge K \\
    0\\le q_i \\le M_i y_i,\\ \forall i;\\ \\ 0\\le q_j \\le M_j z_j,\\ \forall j \\
    \text{{Budget: }} \\sum_i c_i q_i + \\sum_j c_j q_j \\le B.
    """
    "USER QUESTION:\n{input}\n\n"
    "TASK:\n"
    "- Produce a complete LaTeX optimization model using ONLY information in the question.\n"
    "- Use a fixed structure INSIDE LaTeX: Indices/Sets; Given Parameters (convert all tables to arrays); Decision Variables (with domains); Objective; Constraints; Domain lines.\n"
    "- For multi-period problems, you MUST include state-balance recurrences and initial/terminal conditions explicitly.\n"
    "- Avoid nonlinear forms when possible: rewrite ratios/logic using linear constraints + binaries (big-M) with clearly defined M.\n\n"
    "You should decide VARIABLE–TYPE first!\n (If not mentioned or ambiguous, integer by default!)"
    "### FIRST RESPONSE FORMAT (exactly 3 lines) ###\n"
    "Thought: <brief>\n"
    "Action: ORLM_QA\n"
    "Action Input: {input}\n\n"
    "Begin."
)

    suffix = (
    "\n### AFTER OBSERVATION ###\n"
    "Respond with exactly two lines:\n"
    "Thought: <variable types: integer/binary/continuous>\n"
    "Final Answer: <ONLY LaTeX model. Must include: (i) indices/sets, (ii) parameter definitions (tables->arrays), (iii) explicit domain lines for EVERY variable, (iv) initial/terminal conditions if any. No prose.>\n"
    "Do NOT output anything else."
)

    agent = initialize_agent(
        tools=[qa_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": filter_loto_query_only_triggers(prefix),
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=False,  # Enable error handling
    )

    openai.api_request_timeout = 60  
    query = query.replace('{','{{').replace('}','}}')
    output = invoke_react_protocol(lambda: agent, {"input": query}, "Others_without_CSV")["output"]

    return output


# CELL 27
def _legacy_records_from_observation(legacy_observation):
    """Parse legacy evidence only when its structure is unambiguous.

    New runs use planned JSON payloads. This compatibility parser accepts JSON
    arrays/objects, JSON Lines, CSV, and Markdown tables, but rejects ragged or
    otherwise ambiguous input instead of silently constructing malformed records.
    """
    import csv, io

    def normalize_row(row, source):
        if not isinstance(row, dict):
            raise ValueError("Legacy observation rows must be JSON objects or mappings")
        values = row.get("values", row)
        if not isinstance(values, dict) or not values:
            raise ValueError("Legacy observation row has no mapping of values")
        if any(key is None or not str(key).strip() for key in values):
            raise ValueError("Legacy observation contains a missing CSV column name")
        return {"source": source, "values": {str(key).strip(): value for key, value in values.items()}}

    text = str(legacy_observation or "").strip()
    if not text:
        raise ValueError("Legacy observation is empty")

    # Prefer a complete JSON payload, including JSON arrays.
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        if isinstance(decoded, dict) and isinstance(decoded.get("tables"), list):
            records = []
            for table in decoded["tables"]:
                if not isinstance(table, dict):
                    raise ValueError("Legacy observation table must be an object")
                source = str(table.get("source") or table.get("file_name") or "")
                rows = table.get("records")
                if not isinstance(rows, list):
                    raise ValueError("Legacy observation table records must be a list")
                records.extend(normalize_row(row, source) for row in rows)
            if records:
                return records
            raise ValueError("Legacy observation JSON contains no records")
        rows = decoded if isinstance(decoded, list) else [decoded]
        if all(isinstance(row, dict) for row in rows):
            return [normalize_row(row, "") for row in rows]
        raise ValueError("Legacy observation JSON must contain record objects")

    # Split optional source headings, then parse each section strictly.
    sections = re.split(r"(?m)^(?:Retrieved data from )?([\w.-]+\.csv)(?: \(in source order\):)?[ \t]*$", text)
    sections = ["", sections[0], *sections[1:]]
    records = []
    for source, section in zip(sections[::2], sections[1::2]):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines.pop()
        if not lines:
            continue

        # A source heading may precede either a JSON array/object or JSON Lines.
        section_text = "\n".join(lines)
        try:
            section_decoded = json.loads(section_text)
        except json.JSONDecodeError:
            section_decoded = None
        if section_decoded is not None:
            section_rows = section_decoded if isinstance(section_decoded, list) else [section_decoded]
            if not all(isinstance(row, dict) for row in section_rows):
                raise ValueError("Legacy observation JSON section must contain record objects")
            records.extend(normalize_row(row, source) for row in section_rows)
            continue
        if all(line.lstrip().startswith("{") for line in lines):
            try:
                records.extend(normalize_row(json.loads(line), source) for line in lines)
            except json.JSONDecodeError as exc:
                raise ValueError("Legacy JSON Lines observation is malformed") from exc
            continue

        # Markdown table: require a separator row and equal column counts.
        if "|" in lines[0] and len(lines) >= 2 and "-" in lines[1]:
            split_row = lambda line: [part.strip() for part in line.strip().strip("|").split("|")]
            headers, separator = split_row(lines[0]), split_row(lines[1])
            if not headers or len(headers) != len(separator) or any(not h for h in headers):
                raise ValueError("Malformed Markdown table header")
            if any(not re.fullmatch(r":?-{1,}:?", cell) for cell in separator):
                raise ValueError("Malformed Markdown table separator")
            for line in lines[2:]:
                values = split_row(line)
                if len(values) != len(headers):
                    raise ValueError("Ragged Markdown table row")
                records.append({"source": source, "values": dict(zip(headers, values))})
            continue

        reader = csv.DictReader(io.StringIO("\n".join(lines)), skipinitialspace=True)
        if not reader.fieldnames or any(field is None or not field.strip() for field in reader.fieldnames):
            raise ValueError("Legacy CSV observation has invalid column names")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Legacy CSV observation has a ragged row")
            records.append(normalize_row(row, source))
    if not records:
        raise ValueError("Legacy observation format is unsupported or contains no records")
    return records


# ============================================================
# Code generation: shared RAG examples and one prompt builder
# ============================================================

ORIGINAL_CODE_PROMPT = """
You are an expert in mathematical optimization and Python programming.
Write executable Python code that solves the provided mathematical optimization
model with Gurobi, including every decision variable, objective, and constraint.
Preserve the formulation's variable domains. Return code only, without explanations
or Markdown fences.

Mathematical Optimization Model:
{output}
"""

CSV_ROUTE_HINT = {
    "FLP": "Preserve facility-opening and assignment indices and their linking constraints.",
    "AP": "Preserve assignment indices and the matching constraints in the formulation.",
    "TP": "Preserve source-destination axes and supply-demand balance; never assume a square matrix.",
    "RA": """Preserve resource dimensions and capacity constraints. Keep independent
resource-by-activity allocations distinct from global activities consuming multiple
resource dimensions. Index independent allocations by resource, and use each
resource's own decisions or consumption coefficients in its capacity constraint.
Never reuse a global allocation expression against every independent warehouse's
capacity. Preserve all supplied capacities, coefficients, and business identifiers.
Item counts and discrete units must be nonnegative integers unless the problem
explicitly allows fractional quantities or describes divisible material; do not
infer continuity from scale alone.""",
    "NRM": "Preserve product/resource indices and dimension-specific capacity coefficients.",
    "Others": "Follow the formulation exactly; do not introduce unsupported assumptions.",
}

PLANNED_CODE_INSTRUCTIONS = "\nUse the formulation for model structure and Data Mapping. At execution, CSVQA_FRAMES\nis a dictionary from exact table_id to a pandas DataFrame of the original source fields.\nRead every coefficient and entity field from CSVQA_FRAMES[table_id]. Values remain\noriginal strings, including empty fields; explicitly convert required numeric columns.\nDo not redefine these frames, copy coefficients into literals, or read external files.\nUse exact column names and source IDs. Do not guess table/record dictionary layouts.\nCSVQA_DATA remains available ONLY for metadata such as roles and matrix relationships.\nIts top-level table objects have records; values belongs to a record, not to a table.\nDo not access a table as a record or reparse structured source data as CSV text.\nPreserve source row order, complete entity sets and both matrix axes. Select the exact\ntable_id from Data Mapping; roles may repeat. A copied example's entity list never\ndefines the current set without an explicit original-query restriction.\nOptional row_id_mapping/column_id_mapping in matrix metadata maps raw labels to entity\nIDs. Apply it only when supplied; otherwise use exact raw IDs. Preserve all coefficients.\nWithout a continuous pre-horizon value, start change/ramp constraints at the second\nperiod; an initial binary state does not supply a continuous period-zero value.\nOnly parse individual source fields when those fields themselves contain structured text.\n"

LEGACY_CODE_INSTRUCTIONS = """
Build a self-contained program using the concrete coefficients and identifiers in
the Mathematical Optimization Model / Retrieved Information and Original Query.
Define all required data in the code. Do not use CSVQA_DATA, external variables,
or external files.
"""

CSV_SOLVER_INSTRUCTIONS = """
Keep business identifiers as Python dictionary keys, separate from Gurobi names.
Use name='' for addVars/addConstrs so business keys are not expanded into solver names;
use only short ASCII names for individually named variables and constraints.
Use setObjective for the original query's objective sense and full expression, including constants.
Do not substitute an optimizer-equivalent objective or restore omitted constants only in printed output.
Never invent, truncate, or replace required coefficients with dummy or random data.
Before optimization, validate coefficient dimensions and identifier coverage against decision index sets;
raise an explicit error for missing data instead of guessing, padding, or silently dropping entries.
For |linear expression| <= bound, use expression <= bound and expression >= -bound;
do not pass a non-variable expression to addGenConstrAbs.

Gurobi addVars supports lb, ub, obj, vtype and name only; do not invent keyword arguments.
Use a list of unique flat tuple keys with addVars(keys); do not pass a tuplelist and
another Cartesian index unless that extra dimension is intended. Preserve every component
of composite business keys in dictionaries and joins. Sum repeated component rows by the
complete key instead of overwriting them. Ineligible decision pairs must be constrained to
zero or excluded; their presence in source data is not a reason to reject the entire input.
Never use chained comparisons, Python and/or, or bitwise |/& on Gurobi constraints.
Represent logical conditions with explicit binary variables and indicator/linear constraints.
Use pandas Series.str.casefold() for vectorized text and string.casefold() for one string.

Use numeric variable bounds or omit them; never pass None as lb or ub.
Give every decision-variable container a distinct name ending in _vars, such as quantity_vars. Keep scalar keys, loop variables, row objects and parameters separate; never rebind a decision container.
quicksum requires an iterable of numeric or linear terms; use scalar constants directly, never quicksum(0).
Read external CSVs with dtype=str and keep_default_na=False, then explicitly convert required numeric fields. Preserve empty fields and literal identifiers; do not turn a missing relationship into a string 'nan'.
Apply unconditional quantity/resource bounds unconditionally. Only use activation-conditioned bounds when the current query explicitly makes them conditional; an activation fee alone does not make an otherwise unconditional bound conditional.
Add comparisons as constraints; never sum TempConstr objects or use them as linear-expression terms.

Set MIPGap=1e-4 before optimize().
If Status == GRB.OPTIMAL, print ObjVal and every variable's VarName and X.
Otherwise print the solver status; do not report an incumbent as optimal.
"""


def retrieve_csv_code_example(route, query):
    """Render code references from the same structured examples used for modeling."""
    examples = []
    for row in retrieve_rag_examples(normalize_route(route), query, k=2):
        if row["Code"].strip():
            examples.append(
                f"Example {len(examples) + 1}:\n"
                f"Problem:\n{row['prompt']}\n\n"
                f"Formulation:\n{row['Label']}\n\n"
                f"Code:\n{row['Code']}"
            )
    return "\n\n".join(examples)


def _generate_code(output, route, original_query="", data_payload="", legacy_observation=""):
    route = normalize_route(route)
    bind_observation = route == "RA" and bool(legacy_observation) and not data_payload
    parts = [ORIGINAL_CODE_PROMPT.format(output=output)]
    if original_query:
        parts.append(f"Original Query:\n{original_query}")
    if data_payload:
        parts.extend([PLANNED_CODE_INSTRUCTIONS, f"Complete CSVQA_DATA:\n{data_payload}"])
    elif bind_observation:
        records = _legacy_records_from_observation(legacy_observation)
        parts.append(
            "LEGACY_RECORDS is already defined in the program as the exact records below. "
            "Use it directly: each record has 'source' and 'values'; select tables by source "
            "when nonempty, otherwise by fields in record['values']. Do not redefine it, "
            "import it, parse observation text, read files, or copy coefficients into literals. "
            "Derive dimensions and identifiers from these records. Original query determines "
            "semantics and domains; the formulation supplies structure, not guessed data. "
            "Keep explicit per-type capacity bounds; if total capacity denotes their aggregate, "
            "compute their sum rather than inventing another shared limit.\n\n"
            f"Complete LEGACY_RECORDS:\n{json.dumps(records, ensure_ascii=False)}"
        )
    else:
        parts.append(LEGACY_CODE_INSTRUCTIONS)
    parts.append(CSV_ROUTE_HINT[route])

    examples = retrieve_csv_code_example(route, original_query or output)
    if examples:
        parts.extend([
            "The following examples are coding references only. Reuse indexing and "
            "Gurobi modeling patterns, but never copy their coefficients, dimensions, "
            "identifiers, or data. The current formulation and current data are authoritative.",
            examples,
        ])
    parts.extend([CSV_SOLVER_INSTRUCTIONS, MODEL_INTERFACE_INSTRUCTIONS])
    response = make_llm().invoke([HumanMessage(content="\n\n".join(parts))])
    code = response.content
    if bind_observation:
        code = (f"LEGACY_OBSERVATION = {legacy_observation!r}\nLEGACY_RECORDS = {records!r}\n"
                + _source_candidate(code))
    print(code)
    return code


def get_csv_code(output, route, original_query, data_payload="", legacy_observation=""):
    """Generate code with planned data or optional verbatim legacy evidence."""
    return _generate_code(output, route, original_query, data_payload, legacy_observation)


def get_code(output, selected_problem, original_query=""):
    """Generate self-contained code for a query-only formulation."""
    return _generate_code(output, selected_problem, original_query=original_query)


# CELL 29
def _source_candidate(code):
    source = str(code).strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", source, re.DOTALL)
    source = match.group(1) if match else source
    tree = ast.parse(source)
    # Bulk names expand business keys; keep those keys out of Gurobi display names.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"addVars", "addConstrs"}):
            node.keywords = [kw for kw in node.keywords if kw.arg != "name"]
            node.keywords.append(ast.keyword(arg="name", value=ast.Constant(value="")))
    return ast.unparse(tree)


class ModelInterfaceError(RuntimeError):
    """Generated program did not expose a live Gurobi model."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _resolve_generated_model(namespace):
    """Prefer m/model, otherwise accept one distinct global model; never rerun code."""
    for name in ("m", "model"):
        candidate = namespace.get(name)
        if isinstance(candidate, gp.Model):
            return candidate
    candidates = {}
    for name, value in namespace.items():
        if isinstance(value, gp.Model):
            candidates.setdefault(id(value), (value, []))[1].append(name)
    if not candidates:
        types = {name: type(namespace[name]).__name__ for name in ("m", "model") if name in namespace}
        raise ModelInterfaceError(
            "MODEL_INTERFACE_MISSING",
            f"No global Gurobi model was exposed; m/model types: {types}. "
            "A function-local model must be explicitly returned and assigned to m.",
        )
    if len(candidates) != 1:
        names = [aliases for _, aliases in candidates.values()]
        raise ModelInterfaceError("MODEL_INTERFACE_AMBIGUOUS", f"Multiple distinct global Gurobi models: {names}")
    return next(iter(candidates.values()))[0]


def execute_code(code):
    namespace = {"__name__": "__main__"}
    exec(_source_candidate(code), namespace)
    model = _resolve_generated_model(namespace)
    try:
        status = model.Status
    except (gp.GurobiError, AttributeError) as exc:
        raise ModelInterfaceError(
            "MODEL_INTERFACE_DISPOSED", "The exposed model is unavailable; do not dispose it before result extraction."
        ) from exc
    with model:
        if status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"No optimal solution; solver status: {status}")
        objective = model.ObjVal
        solution = [(v.VarName, v.X) for v in model.getVars()]
        print("Optimal objective:", objective)
        print("Optimal solution:", solution)
        return objective, solution


ROUTE_FORMULATORS = {
    "NRM": get_NRM_response, "RA": get_RA_response, "TP": get_TP_response,
    "AP": get_AP_response, "FLP": get_FLP_response,
}


def invoke_formulation(route, query, dataset_address):
    if not dataset_address:
        return {"formulation": get_others_without_CSV_response(query), "trace": {"status": "NOT_APPLICABLE"}}
    return ROUTE_FORMULATORS.get(normalize_route(route), get_Others_response)(query, dataset_address)


def case_fields(case):
    reference = case.get("label_objective")
    address = normalize_data_address(case.get("dataset_address"))
    return {
        "problem_id": str(case["problem_id"]), "query": str(case["Query"]),
        "dataset_address": address, "input_kind": "external_csv" if address else "query_only",
        "true_label": case.get("true_label"), "true_route": case.get("true_route"),
        "label_objective": float(reference) if reference is not None and pd.notna(reference) else None,
    }


def classification_for_case(case):
    return invoke_classifier(case["Query"])


def execute_pipeline_case(case, *, forced_route=None):
    """Classify, formulate, generate code, and solve; retain progress if a step fails."""
    record = {**case_fields(case), "forced_route": forced_route,
              "experiment_mode": "forced" if forced_route else "automatic",
              "repair_count": 0, "retry_count": 0, "fallback_count": 0}
    global REACT_PROTOCOL_DEADLINE
    import time
    REACT_PROTOCOL_EVENTS.clear()
    REACT_PROTOCOL_DEADLINE = time.monotonic() + REACT_PROTOCOL_TIMEOUT_SECONDS
    HTTP_RETRY_EVENTS.clear()
    query, address = record["query"], record["dataset_address"]
    try:
        if forced_route is not None and not address:
            raise ValueError("Forced routes require CSV inputs")
        record["pipeline_stage"] = "classification" if forced_route is None else "formulation"
        classification = classification_for_case(case) if forced_route is None else {}
        label = classification.get("normalized_label")
        route = normalize_route(forced_route) if forced_route is not None else class_to_workflow_route(label)
        record.update(predicted_label=label, assigned_route=route if address else "Others")
        record["pipeline_stage"] = "formulation"
        formulation = invoke_formulation(route, query, address)
        mode = CSVQA_MODE_BY_ROUTE[route] if address else None
        payload = formulation.get("observation", "") if mode == "planned" else ""
        record.update(generated_model=formulation["formulation"], csvqa_observation=formulation.get("observation", ""), csvqa_mode=mode,
                      csvqa_status=formulation.get("trace", {}).get("status"),
                      csvqa_trace=json.dumps(formulation.get("trace", {}), ensure_ascii=False),
                      fallback_count=formulation.get("trace", {}).get("fallback_count", 0))
        model_text = formulation["formulation"]
        if (not isinstance(model_text, str) or not model_text.strip()
                or "agent stopped due to" in model_text.lower()):
            raise RuntimeError("Invalid formulation: empty model or agent stopped before producing a model")
        record["pipeline_stage"] = "code_generation"
        code = formulation.get("code")
        if code is None:
            if address:
                code = get_csv_code(
                    formulation["formulation"], route, query, data_payload=payload,
                    legacy_observation=formulation.get("observation", "")
                    if route == "RA" and mode == "legacy" else "",
                )
            else:
                code = get_code(formulation["formulation"], route, original_query=query)
        source = _source_candidate(code)
        if payload:
            # Include NRM data so the saved program can run on its own.
            data = json.loads(payload) if isinstance(payload, str) else payload
            source = (f"CSVQA_DATA = {pprint.pformat(data, sort_dicts=False, width=120)}\n"
                      "import pandas as pd\n"
                      "CSVQA_FRAMES = {t[\"table_id\"]: pd.DataFrame("
                      "[r[\"values\"] for r in t[\"records\"]], columns=t[\"columns\"], "
                      "index=[r[\"source_row\"] for r in t[\"records\"]]) "
                      "for t in CSVQA_DATA[\"tables\"]}\n" + source)
        record["solve_code"] = source
        record["pipeline_stage"] = "solve"
        objective, solution = execute_code(record["solve_code"])
        record.update(**react_protocol_record_fields())
        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))
        record.update(final_objective=objective, final_solution=solution, final_ok=True,
                      record_status="completed", cache_source="computed", pipeline_stage="completed")
        return record
    except Exception as exc:
        record.update(**react_protocol_record_fields())
        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))
        evidence = getattr(exc, "csvqa_result", {})
        if evidence:
            record.update(csvqa_observation=evidence.get("observation", ""),
                          csvqa_trace=json.dumps(evidence.get("trace", {}), ensure_ascii=False),
                          csvqa_status=evidence.get("trace", {}).get("status"),
                          fallback_count=evidence.get("trace", {}).get("fallback_count", 0))
        record["pipeline_stage"] = getattr(exc, "pipeline_stage", record["pipeline_stage"])
        if isinstance(exc, ModelInterfaceError):
            record["pipeline_stage"] = "result_extraction"
        exc.pipeline_context = record
        raise


# CELL 31
def normalize_data_address(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return "\n".join(line.strip() for line in str(value).splitlines() if line.strip())


def resolve_dataset_address(value, *, dataset_root=None):
    paths = []
    for raw in normalize_data_address(value).splitlines():
        path = Path(raw).expanduser()
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        # An explicit root supports a relocated benchmark directory without editing source data.
        if not resolved.is_file() and dataset_root is not None and not path.is_absolute():
            root = Path(dataset_root).expanduser().resolve()
            candidates = [root / path, root.joinpath(*path.parts[1:])]
            matches = list(dict.fromkeys(candidate.resolve() for candidate in candidates if candidate.is_file()))
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely resolve dataset path {raw!r} under {root}")
            resolved = matches[0]
        paths.append(str(resolved.resolve()))
    return "\n".join(paths)


def parse_label_objective(value):
    text = str(value).replace(",", "").replace("$", "").strip()
    match = re.search(r"Optimal\s+objective\s*([-+\d.eE]+)", text, re.I)
    result = float(match.group(1) if match else text)
    if not math.isfinite(result):
        raise ValueError(f"Reference objective is not finite: {value!r}")
    return result


def prepare_cases(test, *, dataset_root=None):
    """Accept Large-scale-or, IndustryOR, or a DataFrame with at least Query."""
    frame = test.copy()
    if "Query" not in frame:
        raise ValueError("The test dataset must contain a Query column")
    if frame["Query"].isna().any() or frame["Query"].astype(str).str.strip().eq("").any():
        raise ValueError("Every test row must contain a non-empty Query")
    if "problem_id" not in frame:
        frame["problem_id"] = [f"OR-{index + 1:03d}" if isinstance(index, (int, np.integer)) else f"OR-{position:03d}"
                               for position, index in enumerate(frame.index, 1)]
    frame["problem_id"] = frame["problem_id"].astype(str)
    addresses = frame.get("dataset_address", frame.get("Dataset_address", pd.Series("", index=frame.index)))
    frame["dataset_address"] = addresses.map(lambda value: resolve_dataset_address(value, dataset_root=dataset_root))
    labels = frame.get("true_label", frame.get("Problem Type", pd.Series(None, index=frame.index, dtype=object)))
    frame["true_label"] = labels.map(lambda value: normalize_problem_class(value) if pd.notna(value) and str(value).strip() else None)
    frame["true_route"] = frame["true_label"].map(lambda label: class_to_workflow_route(label) if label else None)
    objectives = frame.get("label_objective", frame.get("Label-objective", frame.get("Label", pd.Series(None, index=frame.index, dtype=object))))
    frame["label_objective"] = objectives.map(lambda value: parse_label_objective(value) if pd.notna(value) and str(value).strip() else None)
    return frame


def load_benchmark(path=BENCHMARK_PATH, *, sheet_name=None, dataset_root=None):
    """Read CSV or Excel; an explicit dataset_root relocates relative source paths."""
    path = Path(path).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if path.suffix.lower() not in {".xls", ".xlsx"}:
        return prepare_cases(read_csv_compat(path), dataset_root=dataset_root)
    sheets = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(sheets, pd.DataFrame):
        return prepare_cases(sheets, dataset_root=dataset_root)
    frames = []
    for name, sheet in sheets.items():
        frame = prepare_cases(sheet, dataset_root=dataset_root)
        frame["benchmark_sheet"] = name
        frame["problem_id"] = str(name) + "/" + frame["problem_id"]
        frames.append(frame)
    if not frames:
        raise ValueError(f"No benchmark sheets found in {path}")
    return pd.concat(frames, ignore_index=True)


def objective_is_correct(actual, expected, *, rel_tol=1e-4, abs_tol=1e-4):
    try:
        actual, expected = float(actual), float(expected)
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError, OverflowError):
        return False


def finalize_record(record, *, rel_tol=1e-4, abs_tol=1e-4):
    result = dict(record)
    result["classification_correct"] = (
        result.get("predicted_label") == result["true_label"]
        if result.get("true_label") and result.get("experiment_mode") == "automatic" else None
    )
    reference = result.get("label_objective")
    result["solution_correct"] = (
        bool(result.get("final_ok") and objective_is_correct(result.get("final_objective"), reference, rel_tol=rel_tol, abs_tol=abs_tol))
        if reference is not None and pd.notna(reference) else None
    )
    return result

# CELL 33
def _write_text_atomic(path, text, encoding):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding=encoding, dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


CSV_VALUE_READERS = {
    **dict.fromkeys(("final_ok", "classification_correct", "solution_correct"), lambda value: value == "True"),
    **dict.fromkeys(("final_objective", "label_objective"), float),
}
_RESULT_LOCK = RLock()


def load_records(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return []
    rows = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig").to_dict("records")
    return [{key: CSV_VALUE_READERS.get(key, str)(value) if value else None
             for key, value in row.items()} for row in rows]


def record_key(record):
    return (str(record["problem_id"]), record.get("query", ""),
            normalize_data_address(record.get("dataset_address")), record.get("forced_route") or "AUTO")


RECORD_ARTIFACT_FILES = {"generated_model": "model.md", "solve_code": "solve.py", "csvqa_observation": "data_overview.md", "csvqa_trace": "csvqa_trace.json"}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def materialize_record(csv_path, record):
    """Load a saved record's artifacts so save_record can copy them to another result folder."""
    root, row = Path(csv_path).parent, dict(record)
    for field in RECORD_ARTIFACT_FILES:
        relative = row.pop(field + "_path", None)
        expected_hash = row.pop(field + "_sha256", None)
        artifact = root / relative if relative else None
        if artifact is None or not artifact.is_file() or not expected_hash:
            return None
        try:
            raw = artifact.read_bytes()
            if _sha256_bytes(raw) != expected_hash:
                return None
            row[field] = raw.decode("utf-8")
        except (OSError, UnicodeError):
            return None
    relative = row.pop("solution_path", None)
    expected_hash = row.pop("solution_sha256", None)
    artifact = root / relative if relative else None
    if artifact is None or not artifact.is_file() or not expected_hash:
        return None
    try:
        raw = artifact.read_bytes()
        if _sha256_bytes(raw) != expected_hash:
            return None
        solution = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(solution, dict) or "variables" not in solution:
        return None
    row["final_solution"] = solution["variables"]
    return row


def reusable_record(csv_path, record, fingerprint):
    """Reuse every intact recorded attempt, including failures; never select successes."""
    if (not record or record.get("record_status") not in {"completed", "error"}
            or record.get("cache_fingerprint") != fingerprint):
        return None
    return materialize_record(csv_path, record)


def save_record(csv_path, record):
    """Save readable case files, then atomically update the small results CSV."""
    path = Path(csv_path)
    # Percent-encoding preserves distinct IDs without treating them as paths.
    identity = json.dumps(record_key(record), ensure_ascii=False, separators=(",", ":"), default=str)
    artifact_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    folder = (path.parent / (path.stem + "_cases") / (record.get("forced_route") or "AUTO")
              / quote(str(record["problem_id"]), safe="").replace(".", "%2E") / artifact_id)
    folder.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    with _RESULT_LOCK, path.with_suffix(".csv.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for field, filename in RECORD_ARTIFACT_FILES.items():
            content = row.pop(field, None)
            if content is not None:
                content = str(content)
                _write_text_atomic(folder / filename, content, "utf-8")
                row[field + "_path"] = str((folder / filename).relative_to(path.parent))
                row[field + "_sha256"] = _sha256_bytes(content.encode("utf-8"))
        if "final_solution" in row:
            solution = {"status": "OPTIMAL" if row.get("final_ok") else "ERROR",
                        "objective": row.get("final_objective"), "variables": row.pop("final_solution"),
                        "error": row.get("execution_error")}
            solution_text = json.dumps(solution, ensure_ascii=False, indent=2, allow_nan=False)
            _write_text_atomic(folder / "solution.json", solution_text, "utf-8")
            row["solution_path"] = str((folder / "solution.json").relative_to(path.parent))
            row["solution_sha256"] = _sha256_bytes(solution_text.encode("utf-8"))
        records = [old for old in load_records(path) if record_key(old) != record_key(row)] + [row]
        _write_text_atomic(path, pd.DataFrame(records).to_csv(index=False), "utf-8-sig")
    return records


def _source_fingerprint():
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(
            f"Configured notebook for cache fingerprint does not exist: {NOTEBOOK_PATH}. "
            "Set LEAN_LLM_OPT_NOTEBOOK to the notebook being executed."
        )
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code" and "experiment" not in cell.get("metadata", {}).get("tags", []))
    digest = hashlib.sha256((CACHE_SCHEMA_VERSION + "\n" + str(NOTEBOOK_PATH) + "\n" + source).encode())
    for path in sorted((PROJECT_ROOT / "Large_Scale_Or_Files").rglob("*.csv")):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def case_fingerprint(case, route=None, *, source_fingerprint=None):
    fields = {"cache_schema": CACHE_SCHEMA_VERSION,
              "source": source_fingerprint if source_fingerprint is not None else _source_fingerprint(),
              "model": MODEL_SNAPSHOT,
              "embedding_model": EMBEDDING_MODEL, "modes": CSVQA_MODE_BY_ROUTE,
              "nrm_retry_on_truncation": NRM_RETRY_ON_TRUNCATION,
              "query": str(case["Query"]), "forced_route": route}
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode())
    for raw in normalize_data_address(case.get("dataset_address")).splitlines():
        path = Path(raw)
        digest.update(str(path).encode())
        digest.update(path.read_bytes() if path.is_file() else b"MISSING_FILE")
    return digest.hexdigest()


def error_record(case, exc, forced_route=None):
    return {
        **case_fields(case), "assigned_route": forced_route, "predicted_label": None,
        **getattr(exc, "pipeline_context", {}),
        "experiment_mode": "forced" if forced_route else "automatic", "forced_route": forced_route,
        "final_ok": False, "final_objective": None, "final_solution": None,
        "execution_error": f"{type(exc).__name__}: {exc}", "execution_error_type": type(exc).__name__,
        "execution_error_code": getattr(exc, "code", None),
        "record_status": "error", "cache_source": "computed",
        "generated_model": getattr(exc, "pipeline_context", {}).get("generated_model", ""),
        "solve_code": getattr(exc, "pipeline_context", {}).get("solve_code", ""),
        "csvqa_observation": getattr(exc, "pipeline_context", {}).get("csvqa_observation", ""),
        "csvqa_trace": getattr(exc, "pipeline_context", {}).get("csvqa_trace", "{}"),
        "repair_count": 0, **react_protocol_record_fields(),
        "fallback_count": getattr(exc, "pipeline_context", {}).get("fallback_count", 0),
    }


# CELL 35
def run_test(test, *, output_csv=RESULTS_DIR / "results.csv", routes=None, reuse_auto_csv=None,
             reuse_completed=True, continue_on_error=True, rel_tol=1e-4, abs_tol=1e-4):
    """Evaluate once per case/version; resume recorded successes and failures."""
    frame = load_benchmark(test) if isinstance(test, (str, Path)) else prepare_cases(test)
    selected_routes = [None] if routes is None else [normalize_route(route) for route in routes]
    if not selected_routes or len(selected_routes) != len(set(selected_routes)):
        raise ValueError("Provide a non-empty list of distinct routes")
    if routes is not None and frame["dataset_address"].eq("").any():
        raise ValueError("Forced-route tests require case CSV inputs; use routes=None for query-only tests")
    existing = load_records(output_csv)
    if existing and not reuse_completed:
        raise ValueError("Use a new round directory for a fresh run; historical attempts cannot be overwritten")
    cached = {record_key(row): row for row in existing}
    auto_cached = ({record_key(row): row for row in load_records(reuse_auto_csv)}
                   if reuse_completed and reuse_auto_csv else {})
    # Fingerprint every run before saving records so a first run is reusable later.
    source = _source_fingerprint()
    results, api_checked = [], False
    total = len(frame) * len(selected_routes)
    for case in frame.to_dict("records"):
        for route in selected_routes:
            key = (case["problem_id"], str(case["Query"]), case["dataset_address"], route or "AUTO")
            fingerprint = case_fingerprint(case, route, source_fingerprint=source)
            if key in cached and cached[key].get("cache_fingerprint") != fingerprint:
                raise ValueError("Existing attempts use different code/data; choose a new version directory")
            record = reusable_record(output_csv, cached.get(key), fingerprint)
            if key in cached and record is None:
                raise ValueError("Recorded attempt has missing/corrupt artifacts; do not rerun it")
            if record is not None:
                record["cache_source"] = "csv"
            elif route is not None and route == case.get("true_route"):
                automatic = auto_cached.get(key[:-1] + ("AUTO",))
                if automatic and automatic.get("assigned_route") == route:
                    auto_fingerprint = case_fingerprint(case, source_fingerprint=source)
                    record = reusable_record(reuse_auto_csv, automatic, auto_fingerprint)
                    if record is not None:
                        record["cache_source"] = "oracle_auto_reuse"
            if record is not None:
                record.update(**case_fields(case), forced_route=route,
                              experiment_mode="forced" if route else "automatic")
            else:
                # A missing credential is a run configuration error, not 101 case failures.
                if not api_checked:
                    require_api_key()
                    api_checked = True
                try:
                    record = execute_pipeline_case(case, forced_route=route)
                except Exception as exc:
                    record = error_record(case, exc, route)
                    print(f"{case['problem_id']} / {route or 'AUTO'} "
                          f"failed at {record.get('pipeline_stage', 'setup')}: {record['execution_error']}")
                    if not continue_on_error:
                        record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
                        record["cache_fingerprint"] = fingerprint
                        save_record(output_csv, record)
                        raise
            record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
            record["cache_fingerprint"] = fingerprint
            record["base_notebook_sha256"] = globals().get("BASE_NOTEBOOK_SHA256", hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest())
            save_record(output_csv, record)
            results.append(record)
            print(f"[{len(results)}/{total}] {case['problem_id']} / {route or 'AUTO'}: "
                  f"{record['record_status']} [{record.get('cache_source')}]")
    return results


def evaluation_summary(records):
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["metric", "correct", "total", "accuracy"])
    rows = []
    if frame["experiment_mode"].eq("forced").all():
        oracle = frame.loc[
            frame["forced_route"].eq(frame["true_route"])
            & frame["solution_correct"].notna()
        ].drop_duplicates("problem_id")
        for label, correct, total in (
                ("Oracle routing accuracy", int(oracle["solution_correct"].eq(True).sum()), oracle["problem_id"].nunique()),
                ("Solved accuracy", int(frame["final_ok"].eq(True).sum()), len(frame))):
            rows.append({"metric": label, "correct": correct, "total": total,
                         "accuracy": correct / total if total else None})
        return pd.DataFrame(rows)
    for field, label in (("final_ok", "Solved"), ("classification_correct", "Classification"), ("solution_correct", "Objective match")):
        values = frame[field].dropna() if field in frame else pd.Series(dtype=bool)
        total, correct = len(values), int(values.eq(True).sum())
        rows.append({"metric": label, "correct": correct, "total": total,
                     "accuracy": correct / total if total else None})
    return pd.DataFrame(rows)


def classification_confusion(records):
    frame = pd.DataFrame(records)
    frame = frame.loc[frame["true_label"].notna() & frame["experiment_mode"].eq("automatic")]
    predicted = frame["predicted_label"].fillna("Unclassified")
    columns = CLASS_LABELS + (["Unclassified"] if predicted.eq("Unclassified").any() else [])
    return pd.crosstab(frame["true_label"], predicted).reindex(index=CLASS_LABELS, columns=columns, fill_value=0)


def accuracy_by_class(records):
    frame = pd.DataFrame(records)
    grouped = frame.groupby("true_label").agg(
        total=("problem_id", "size"), classification_correct=("classification_correct", "sum"),
        classification_scored=("classification_correct", "count"), solution_correct=("solution_correct", "sum"),
        solution_scored=("solution_correct", "count"))
    for metric in ("classification", "solution"):
        grouped[metric + "_accuracy"] = grouped[metric + "_correct"] / grouped[metric + "_scored"].replace(0, np.nan)
    return grouped.reindex(CLASS_LABELS).rename_axis("true_label")


def forced_route_matrix(records):
    frame = pd.DataFrame(records)
    grouped = frame.groupby(["true_route", "assigned_route"])["solution_correct"].agg(correct="sum", total="count")
    formatted = grouped.apply(lambda row: f"{int(row.correct)}/{int(row.total)} ({row.correct / row.total:.2%})" if row.total else "Not scored", axis=1)
    return formatted.unstack("assigned_route").reindex(index=WORKFLOW_ROUTES, columns=WORKFLOW_ROUTES)


def accuracy_by_route(records):
    grouped = pd.DataFrame(records).groupby("assigned_route")["solution_correct"].agg(correct="sum", total="count")
    grouped["accuracy"] = grouped["correct"] / grouped["total"].replace(0, np.nan)
    return grouped.reindex(WORKFLOW_ROUTES)


def report_results(records, output_dir):
    """Display and save the tables appropriate to the experiment."""
    tables = {"summary": evaluation_summary(records)}
    if records:
        if all(row.get("experiment_mode") == "automatic" for row in records):
            tables.update(classification_confusion=classification_confusion(records), accuracy_by_class=accuracy_by_class(records))
        else:
            tables.update(route_matrix=forced_route_matrix(records), accuracy_by_route=accuracy_by_route(records))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        print(name.replace("_", " ").title())
        shown = table.copy()
        for column in shown.columns:
            if str(column).endswith("accuracy"):
                shown[column] = shown[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "Not scored")
        display(shown)
        table.to_csv(output_dir / f"{name}.csv", index=table.index.name is not None, encoding="utf-8-sig")
    return tables


# CELL 37
# The original model-specific functions above stay intact; these controls scope each fold.
LOTO_STATE = None
_LOTO_BASE_CLASSIFIER_PREFIX = getattr(globals().get("_loto_classifier_prefix"), "base_prefix", prefix)
_LOTO_BASE_CASE_FINGERPRINT = getattr(case_fingerprint, "_loto_base", case_fingerprint)
_LOTO_BASE_INVOKE_FORMULATION = getattr(invoke_formulation, "_loto_base", invoke_formulation)
_LOTO_BASE_EXECUTE_PIPELINE = getattr(execute_pipeline_case, "_loto_base", execute_pipeline_case)
_LOTO_BASE_FINALIZE_RECORD = getattr(finalize_record, "_loto_base", finalize_record)


class DisabledRouteError(ValueError):
    """A classifier selected a workflow that is unavailable in the current fold."""


def require_loto_fold():
    if LOTO_STATE is None:
        raise RuntimeError("Select a leave-one-type-out fold before running inference")
    return LOTO_STATE


def loto_allowed_labels():
    return list(require_loto_fold()["allowed_labels"])


def loto_reference_frame():
    """Return the filtered rows before Mixture/Others are merged into a workflow route."""
    return require_loto_fold()["reference_frame"].copy()


def _loto_classifier_prefix():
    state = require_loto_fold()
    blocks = re.split(r"(?=Example \d+)", few_shot_example)
    kept = [block for block in blocks if not re.search(
        r"Final Answer:\s*" + re.escape(state["held_out_type"]) + r"\b", block)]
    fold_prefix = _LOTO_BASE_CLASSIFIER_PREFIX.replace(few_shot_example, "".join(kept))
    if not state["disabled_route"]:
        return fold_prefix
    # Keep original semantic definitions and hand-written examples. Only availability changes.
    tokens = ", ".join(state["allowed_labels"])
    return fold_prefix + f"""

WORKFLOW AVAILABILITY FOR THIS RUN (overrides the earlier output-label list):
The only available Final Answer labels are: {tokens}.
Some examples and definitions above describe unavailable workflows. They remain
reference material, but their labels are not valid choices for this run.
Select the most appropriate AVAILABLE workflow from the current problem's structure.
Do not output any other label. Do not infer the current problem's true label from
which workflows are available. Keep the same one-FileQA-call ReAct protocol.
"""


_loto_classifier_prefix.base_prefix = _LOTO_BASE_CLASSIFIER_PREFIX


def set_loto_fold(held_out_type):
    global LOTO_STATE, prefix
    semantic_type = normalize_problem_class(held_out_type)
    if semantic_type not in CLASS_LABELS:
        raise ValueError(f"Unsupported held-out semantic type: {held_out_type!r}")
    path = Path(RAG_EXAMPLES_ALL_PATH).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    frame = read_csv_compat(path, dtype=str, keep_default_na=False)
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    if required.difference(frame.columns):
        raise ValueError(f"RAG example CSV missing fields: {sorted(required.difference(frame.columns))}")
    semantic_labels = frame["Type"].map(normalize_problem_class)
    remove = semantic_labels.eq(semantic_type)
    filtered = frame.loc[~remove].copy()
    if len(filtered) < 5:
        raise ValueError("At least five remaining examples are required by the original FileQA protocol")
    disabled_route = class_to_workflow_route(semantic_type) if LOTO_REMOVE_ROUTE else None
    allowed = [label for label in CLASS_LABELS if class_to_workflow_route(label) != disabled_route]
    LOTO_STATE = {
        "held_out_type": semantic_type, "disabled_route": disabled_route,
        "allowed_labels": allowed, "reference_frame": filtered,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_path": str(path.resolve()),
        "removed_example_indices": [int(index) for index in frame.index[remove]],
        "removed_examples": [
            {"row_index": int(index), "type": str(frame.loc[index, "Type"]),
             "prompt": str(frame.loc[index, "prompt"])} for index in frame.index[remove]
        ],
        "remaining_example_indices": [int(index) for index in filtered.index],
        "remaining_counts": filtered["Type"].map(normalize_problem_class).value_counts().to_dict(),
        "reference_count_before": len(frame), "reference_count_after": len(filtered),
    }
    # Reset every reference/agent cache at fold boundaries, including the shared Others store.
    for name in ("_get_classifier_retriever", "_get_classification_agent", "_load_rag_table",
                 "_get_rag_store", "_unified_rag_rows", "get_route_retriever", "get_others_store"):
        clear = getattr(globals().get(name), "cache_clear", None)
        if clear is not None:
            clear()
    prefix = _loto_classifier_prefix()
    return loto_manifest()


def loto_manifest():
    state = require_loto_fold()
    return {
        "experiment": LOTO_VARIANT, "model": MODEL_SNAPSHOT,
        "base_notebook": LOTO_BASE_NOTEBOOK, "base_notebook_sha256": LOTO_BASE_SHA256,
        "held_out_type": state["held_out_type"], "disabled_workflow_route": state["disabled_route"],
        "allowed_labels": state["allowed_labels"],
        "allowed_routes": [route for route in WORKFLOW_ROUTES if route != state["disabled_route"]],
        "reference_path": state["reference_path"], "reference_sha256": state["reference_sha256"],
        "reference_count_before": state["reference_count_before"],
        "reference_count_after": state["reference_count_after"],
        "removed_examples": state["removed_examples"],
        "remaining_example_indices": state["remaining_example_indices"],
        "remaining_counts": state["remaining_counts"],
        "removal_scope": "Exact semantic Type in RAG_Examples_All; UFLP normalized to FLP",
        "fixed_prompt_examples": "Target-type classifier demonstrations removed; generic definitions retained",
        "classification": "Recomputed using the current fold; no external classification cache",
    }


def loto_record_fields():
    state = require_loto_fold()
    return {"loto_variant": LOTO_VARIANT, "held_out_type": state["held_out_type"],
            "disabled_workflow_route": state["disabled_route"],
            "allowed_labels": ",".join(state["allowed_labels"]),
            "reference_sha256": state["reference_sha256"],
            "reference_count_after": state["reference_count_after"]}


def assert_loto_route_allowed(route):
    state = require_loto_fold()
    route = normalize_route(route)
    if route == state["disabled_route"]:
        raise DisabledRouteError(f"Workflow {route} is disabled in the {state['held_out_type']} fold")
    return route


def invoke_formulation(route, query, dataset_address):
    # Explicit guard prevents .get(..., get_Others_response) from bypassing a disabled route.
    assert_loto_route_allowed(route if dataset_address else "Others")
    return _LOTO_BASE_INVOKE_FORMULATION(route, query, dataset_address)


invoke_formulation._loto_base = _LOTO_BASE_INVOKE_FORMULATION


def execute_pipeline_case(case, *, forced_route=None):
    require_loto_fold()
    if forced_route is not None:
        raise ValueError("LOTO inference must reclassify; forced routes are not supported")
    try:
        record = _LOTO_BASE_EXECUTE_PIPELINE(case, forced_route=None)
    except Exception as exc:
        context = {**getattr(exc, "pipeline_context", {}), **loto_record_fields()}
        if isinstance(exc, DisabledRouteError):
            context.update(pipeline_stage="route_selection", route_allowed=False)
        exc.pipeline_context = context
        raise
    record.update(**loto_record_fields(), route_allowed=True)
    return record


execute_pipeline_case._loto_base = _LOTO_BASE_EXECUTE_PIPELINE


def finalize_record(record, *, rel_tol=1e-4, abs_tol=1e-4):
    result = _LOTO_BASE_FINALIZE_RECORD(record, rel_tol=rel_tol, abs_tol=abs_tol)
    # Gold-label accuracy remains measurable; the forbidden label is expected to score zero.
    return result


finalize_record._loto_base = _LOTO_BASE_FINALIZE_RECORD


def case_fingerprint(case, route=None, *, source_fingerprint=None):
    baseline = _LOTO_BASE_CASE_FINGERPRINT(case, route, source_fingerprint=source_fingerprint)
    identity = {**loto_record_fields(), "baseline": baseline,
                "ollama_base_url": globals().get("OLLAMA_BASE_URL"),
                "num_ctx": os.environ.get("LEAN_NUM_CTX", "131072"),
                "num_predict": os.environ.get("LEAN_NUM_PREDICT", "8192"),
                "seed": os.environ.get("LEAN_SEED", "42")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


case_fingerprint._loto_base = _LOTO_BASE_CASE_FINGERPRINT


# CELL 39
@lru_cache(maxsize=1)
def _get_classifier_retriever():
    frame = loto_reference_frame()
    documents = [
        Document(
            page_content="\n".join(
                f"{column}: {str(example[column]).strip()}"
                for column in ("prompt", "Type")
            ),
            metadata={
                "source": str(PROJECT_ROOT / "Large_Scale_Or_Files/RAG_Examples_All.csv"),
                "row": row_number,
            },
        )
        for row_number, example in enumerate(frame.to_dict("records"))
    ]
    if len(documents) < 5:
        raise ValueError("Classification reference requires at least five labeled examples")
    return FAISS.from_documents(documents, make_embeddings()).as_retriever(
        search_kwargs={"k": 5}
    )


@lru_cache(maxsize=1)
def _load_rag_table():
    """Read the filtered examples once, preserving every field as text."""
    frame = loto_reference_frame().copy()
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RAG example CSV is missing columns: {sorted(missing)}")
    frame["Type"] = frame["Type"].map(normalize_route)
    return frame


def load_rag_examples(route):
    """Return one canonical route; UFLP maps to FLP and Mixture to Others."""
    frame = _load_rag_table()
    examples = frame.loc[frame["Type"].eq(normalize_route(route))].copy()
    return examples


def retrieve_rag_examples(route, query, k=1):
    """Retrieve structured example dictionaries, never parse CSVLoader display text."""
    if k < 1:
        return []
    route = normalize_route(route)
    examples = load_rag_examples(route)
    if examples.empty:
        return []
    documents = _get_rag_store(route).similarity_search(
        str(query), k=min(k, len(examples)),
    )
    return [json.loads(document.metadata["example_json"]) for document in documents]



def filter_loto_query_only_documents(documents):
    """Scope the separate query-only library by its original problem-type field."""
    target = require_loto_fold()["held_out_type"]
    def semantic_type(document):
        match = re.search(r"^problem type:\s*(.*)$", document.page_content, re.MULTILINE | re.I)
        text = match.group(1).casefold() if match else "others"
        if "network revenue" in text:
            return "NRM"
        if "resource allocation" in text or "knapsack" in text:
            return "RA"
        if "assignment" in text:
            return "AP"
        if "facility location" in text:
            return "FLP"
        if "transport" in text or "transshipment" in text or "minimum-cost flow" in text:
            return "TP"
        return "Others"
    return [document for document in documents if semantic_type(document) != target]


def filter_loto_query_only_triggers(text):
    """Remove target-type fixed demonstrations while retaining generic contracts."""
    target = require_loto_fold()["held_out_type"]
    # Explicit structural types of the three fixed baseline demonstrations.
    trigger_types = {1: "RA", 2: "Mixture", 3: "Mixture"}
    for number, label in trigger_types.items():
        if label == target:
            text = re.sub(r"\(" + str(number) + r"\).*?(?=\(\d\)|USER QUESTION:)",
                          "", text, flags=re.DOTALL)
    return text


# CELL 41
def loto_preflight():
    """Validate the seven disjoint gold-label folds and CSV paths without model calls."""
    frame = load_benchmark()
    if len(frame) != 101 or frame["problem_id"].duplicated().any():
        raise ValueError("Expected 101 cases with unique problem IDs")
    if frame["true_label"].isna().any() or not frame["true_label"].isin(CLASS_LABELS).all():
        raise ValueError("Every LOTO case needs a supported gold semantic label for grouping")
    for address in frame["dataset_address"]:
        paths = normalize_data_address(address).splitlines()
        if not paths:
            raise ValueError("These notebooks evaluate only the 101 cases with CSV inputs")
        for raw in paths:
            if not Path(raw).is_file():
                raise FileNotFoundError(raw)
    rows = []
    for held_out in CLASS_LABELS:
        manifest = set_loto_fold(held_out)
        rows.append({"held_out_type": held_out, "cases": int(frame["true_label"].eq(held_out).sum()),
                     "examples_removed": len(manifest["removed_examples"]),
                     "examples_remaining": manifest["reference_count_after"],
                     "disabled_route": manifest["disabled_workflow_route"],
                     "allowed_labels": ",".join(manifest["allowed_labels"])})
    preview = pd.DataFrame(rows)
    if preview["cases"].sum() != 101 or preview["cases"].eq(0).any():
        raise ValueError("LOTO folds must partition all 101 cases without empty types")
    return frame, preview


def loto_report(records, output_dir):
    frame = pd.DataFrame(records)
    if "pipeline_stage" not in frame:
        frame["pipeline_stage"] = "setup"
    else:
        frame["pipeline_stage"] = frame["pipeline_stage"].fillna("setup")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for held_out, group in frame.groupby("held_out_type", sort=False):
        total = len(group)
        rows.append({"held_out_type": held_out, "cases": total,
                     "solved": int(group["final_ok"].eq(True).sum()),
                     "objective_correct": int(group["solution_correct"].eq(True).sum()),
                     "solve_rate": float(group["final_ok"].eq(True).mean()),
                     "objective_accuracy": float(group["solution_correct"].eq(True).mean())})
    per_type = pd.DataFrame(rows)
    summary_rows = []
    for label, field in [("Solved", "final_ok"), ("Objective match", "solution_correct")]:
        count = int(frame[field].eq(True).sum())
        summary_rows.append({"metric": label, "correct": count, "total": len(frame),
                             "accuracy": count / len(frame)})
    summary_rows.append({"metric": "Classification", "correct": int(frame["classification_correct"].eq(True).sum()),
                         "total": len(frame), "accuracy": float(frame["classification_correct"].eq(True).mean())})
    summary_rows.append({"metric": "Macro objective match across evaluated types", "correct": None,
                         "total": len(per_type), "accuracy": float(per_type["objective_accuracy"].mean())})
    summary = pd.DataFrame(summary_rows)
    routes = frame.groupby(["held_out_type", "assigned_route"], dropna=False).size().rename("cases").reset_index()
    failures = frame.loc[~frame["final_ok"].eq(True)].groupby(
        ["held_out_type", "pipeline_stage", "execution_error_type"], dropna=False
    ).size().rename("cases").reset_index() if "execution_error_type" in frame else pd.DataFrame()
    columns = [column for column in (
        "problem_id", "held_out_type", "loto_variant", "true_label", "predicted_label",
        "assigned_route", "disabled_workflow_route", "allowed_labels", "route_allowed",
        "final_ok", "final_objective", "label_objective", "solution_correct", "classification_correct",
        "pipeline_stage", "execution_error_type", "execution_error", "cache_source", "cache_fingerprint",
    ) if column in frame]
    tables = {"summary": summary, "accuracy_by_held_out_type": per_type,
              "selected_routes": routes, "failures": failures, "case_summary": frame[columns]}
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    display(per_type)
    display(summary)
    return tables


def run_loto(*, folds=None, rows=None, reuse_completed=True, continue_on_error=True,
             rel_tol=1e-4, abs_tol=1e-4):
    """Fresh classification per case/fold; resume only matching, complete successful results."""
    frame, _ = loto_preflight()
    chosen = list(CLASS_LABELS) if folds is None else [normalize_problem_class(value) for value in folds]
    if not chosen or len(chosen) != len(set(chosen)) or any(value not in CLASS_LABELS for value in chosen):
        raise ValueError("FOLDS must contain distinct supported semantic types")
    if rows is not None:
        rows = list(rows)
        if (len(rows) != len(set(rows)) or any(isinstance(row, (bool, np.bool_))
                or not isinstance(row, (int, np.integer)) or not 0 <= row < len(frame) for row in rows)):
            raise ValueError("ROWS must be distinct zero-based benchmark row indices from 0 to 100")
        frame = frame.iloc[rows]
    selected = frame.loc[frame["true_label"].isin(chosen)]
    if selected.empty:
        raise ValueError("FOLDS and ROWS select no cases")
    source = _source_fingerprint()
    results, api_checked = [], False
    for held_out in chosen:
        fold_cases = selected.loc[selected["true_label"].eq(held_out)]
        if fold_cases.empty:
            continue
        manifest = set_loto_fold(held_out)
        folder = RESULTS_DIR / held_out
        folder.mkdir(parents=True, exist_ok=True)
        manifest.update(case_ids=fold_cases["problem_id"].tolist(), source_fingerprint=source,
                        rel_tol=rel_tol, abs_tol=abs_tol)
        _write_text_atomic(folder / "fold_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        output_csv = folder / "results.csv"
        existing = load_records(output_csv)
        if existing and not reuse_completed:
            raise ValueError("Use a fresh round directory; recorded LOTO attempts cannot be overwritten")
        cached = {record_key(record): record for record in existing}
        for case in fold_cases.to_dict("records"):
            fingerprint = case_fingerprint(case, source_fingerprint=source)
            key = (case["problem_id"], str(case["Query"]), case["dataset_address"], "AUTO")
            previous = cached.get(key)
            if previous and previous.get("cache_fingerprint") != fingerprint:
                raise ValueError("LOTO code/data changed; use a fresh version directory")
            record = None
            if (previous and previous.get("record_status") in {"completed", "error"}
                    and previous.get("cache_fingerprint") == fingerprint):
                record = materialize_record(output_csv, previous)
                if record is not None:
                    record.update(cache_source="csv", **loto_record_fields())
            if record is None:
                if not api_checked and callable(globals().get("require_api_key")):
                    require_api_key()
                api_checked = True
                try:
                    record = execute_pipeline_case(case)
                except Exception as exc:
                    record = error_record(case, exc)
                    record.update(loto_record_fields())
                    if not continue_on_error:
                        record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
                        record["cache_fingerprint"] = fingerprint
                        save_record(output_csv, record)
                        raise
            record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
            record["cache_fingerprint"] = fingerprint
            save_record(output_csv, record)
            results.append(record)
            print(f"[{len(results)}/{len(selected)}] {held_out} / {case['problem_id']}: "
                  f"{record['record_status']} via {record.get('assigned_route')} [{record.get('cache_source')}]")
    loto_report(results, RESULTS_DIR)
    return results


def run_test(*args, **kwargs):
    raise RuntimeError("Use run_loto(folds=..., rows=...) to preserve fold and cache isolation")


# CELL 43
# Run loto_preflight() explicitly before enabling the experiment.


# CELL 45
RUN_LOTO = False
FOLDS = None
ROWS = None
REUSE_COMPLETED = False
if RUN_LOTO:
    loto_results = run_loto(folds=FOLDS, rows=ROWS, reuse_completed=REUSE_COMPLETED)
