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
    "Retrieve every CSV row needed by the query in source order. Preserve exact identifiers and "
    "numeric values. If the query names a subset, retrieve that complete subset; otherwise retrieve "
    "all data. Use only the supplied context: {context}"
)


CSVQA_LEGACY_TOOL_DESCRIPTION = (
    "Retrieve complete source-ordered CSV evidence needed to formulate the optimization problem."
)

def invoke_react_with_required_csvqa(agent, query, route):
    """Require CSVQA while preserving the existing ReAct agent boundary."""
    def call_count(result):
        return sum(
            isinstance(step, tuple)
            and len(step) >= 2
            and getattr(step[0], "tool", None) == "CSVQA"
            for step in result.get("intermediate_steps", [])
        )

    result = agent.invoke(query)
    if call_count(result) == 0:
        result = agent.invoke(
            "Protocol correction: call CSVQA before the Final Answer. Begin with Action: CSVQA.\n\n"
            f"User Description: {query}"
        )
    calls = call_count(result)
    planned = CSVQA_MODE_BY_ROUTE[normalize_route(route)] == "planned"
    if (planned or normalize_route(route) == "RA") and calls != 1:
        raise RuntimeError(f"{route} ReAct must call CSVQA exactly once; observed {calls}")
    if not planned and calls < 1:
        raise RuntimeError(f"{route} legacy ReAct must call CSVQA at least once; observed {calls}")
    return result


CSVQA_PLANNED_PROMPTS = {'NRM': 'Select every source row and column needed to define the revenue-management objective, decision entities, demand limits, inventory or capacity limits, and query-specific bounds. Infer exact columns from the query and schema. Filter only an explicit subset.'}

CSVQA_TOOL_DESCRIPTIONS = {'NRM': 'Return the canonical CSV rows, columns, filters, and relationships needed by the query.'}

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


def _nonempty_cell(value):
    return pd.notna(value) and str(value).strip() != ""


def _repeated_header_mask(frame):
    """Find repeated headers and one-cell matrix axis-title rows generically."""
    key = lambda value: "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
    columns = [key(column) for column in frame.columns]

    def is_repeated_header(row):
        comparable = [
            (columns[index], key(value))
            for index, value in enumerate(row.tolist())
            if _nonempty_cell(value)
        ]
        if len(comparable) >= 2:
            matches = sum(bool(column) and column == value for column, value in comparable)
            return matches / len(comparable) >= 0.8
        if len(comparable) != 1 or not _nonempty_cell(row.iloc[0]):
            return False
        later = frame.loc[frame.index > row.name]
        if later.empty:
            return False
        later_nonempty = later.apply(
            lambda values: values.map(_nonempty_cell).sum(), axis=1
        )
        dense_later = float(later_nonempty.ge(max(2, len(frame.columns) // 2)).mean()) >= 0.8
        later_first_numeric = float(
            pd.to_numeric(later.iloc[:, 0], errors="coerce").notna().mean()
        ) >= 0.8
        return dense_later and later_first_numeric

    return frame.apply(is_repeated_header, axis=1)


def _clean_csv_frame(frame):
    """Remove only empty structure, repeated headers, and clear export indices."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    frame = frame.copy()
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    nonempty = frame.apply(lambda column: column.map(_nonempty_cell))
    frame = frame.loc[nonempty.any(axis=1), nonempty.any(axis=0)].copy()
    repeated_headers = _repeated_header_mask(frame)
    if repeated_headers.any():
        frame = frame.loc[~repeated_headers].copy()
    for column in list(frame.columns):
        values = frame[column].astype(str).tolist()
        sequential = values in (
            [str(i) for i in range(len(values))],
            [str(i) for i in range(1, len(values) + 1)],
        )
        if sequential and re.fullmatch(r"(?:unnamed(?::\s*\d+)?|index)", column.strip(), re.I):
            frame = frame.drop(columns=column)
    return frame


def _read_csv(path):
    frame = read_csv_compat(path, dtype=str, keep_default_na=False)
    return _clean_csv_frame(frame)


def _load_tables(dataset_address):
    """Resolve one-or-more newline-separated CSV paths in source order."""
    tables = []
    for raw in str(dataset_address or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSVQA cannot find dataset file: {path}")
        tables.append({"file_index": len(tables), "path": path, "frame": _read_csv(path)})
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
