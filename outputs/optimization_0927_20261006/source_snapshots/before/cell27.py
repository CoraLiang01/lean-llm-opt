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

PLANNED_CODE_INSTRUCTIONS = """
Use the formulation for model structure and Data Mapping; use CSVQA_DATA for all data.
The complete payload below is provided at execution as CSVQA_DATA. Do not define
or overwrite it, copy its rows into literals, or read external files.
Select tables by the exact table_id in Data Mapping; roles may repeat.
Derive dimensions and identifiers from CSVQA_DATA["tables"][i]["records"], reading
fields through record["values"][column_name]. Preserve table mapping, record order,
identifiers, and matrix axes. Do not read CSVQA_DATA["plan"].
Without a continuous pre-horizon value, start change/ramp constraints at the second
period; an initial binary state does not supply a continuous period-zero value.
Parse structured CSV text with a regex and fail explicitly if any clause is unmatched.
"""

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
Leave the solved Gurobi model in m or model. If using a function, return the model
and assign it to m. Set MIPGap=1e-4 before optimize().
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
    parts.append(CSV_SOLVER_INSTRUCTIONS)
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
