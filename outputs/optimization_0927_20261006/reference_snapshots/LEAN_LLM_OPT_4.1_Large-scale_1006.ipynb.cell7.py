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
