"""Build a review candidate without modifying the delivered notebooks."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
BASE = OUT / "baseline/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb"
DEST = OUT / "candidates/full_review_v2.ipynb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_function(source, name, replacement):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return "".join(lines[:start]) + replacement.strip() + "\n\n" + "".join(lines[node.end_lineno:])


def main():
    book = json.loads(BASE.read_text())
    changed = []

    def edit(index, old, new):
        source = "".join(book["cells"][index]["source"])
        assert source.count(old) == 1, (index, old[:80])
        book["cells"][index]["source"] = source.replace(old, new).splitlines(True)
        changed.append({"cell": index, "old": old, "new": new})

    source = "".join(book["cells"][7]["source"])
    replacement = '''def _read_csv(path):
    """Preserve every parsed source row, column and text value, including index-like IDs."""
    frame = read_csv_compat(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).lstrip("\\ufeff") for column in frame.columns]
    return frame
'''
    book["cells"][7]["source"] = replace_function(source, "_read_csv", replacement).splitlines(True)
    changed.append({"cell": 7, "function": "_read_csv", "reason": "Sequential index-like columns and blank axes may be business data."})
    edit(7, 'value_matches[str(column)] = counts',
         'value_matches[str(column)] = {**counts, "matching_values": frame.loc[values.str.contains(wanted, regex=False), column].drop_duplicates().head(4).tolist()}')
    edit(9, 'Every explicit subset must have a filter and every filter must quote supporting query evidence.',
         'Every explicit subset must have a filter and every filter must quote supporting query evidence.\nFor a query-named category fragment or family, inspect actual matching_values and exact/prefix/contains\ncounts before choosing the supported predicate. Do not equate a fragment to a complete source label\nwhen the query describes a family and the profile shows no exact match. Preserve literal identifiers.')
    edit(9, '    value = condition.get("value")\n    wanted =',
         '    if operator in {"prefix", "contains"} and dtype != "string":\n        raise ValueError("prefix/contains require string data")\n    value = condition.get("value")\n    if isinstance(value, list) and operator not in {"in", "not_in", "between"}:\n        raise ValueError("Scalar operators require a scalar value")\n    if value is None:\n        raise ValueError("Non-null predicates require a value")\n    wanted =')
    edit(9, '    key = lambda value: "".join(ch for ch in str(value or "").casefold() if ch.isalnum())',
         '    key = lambda value: re.sub(r"\\s+", " ", str(value)).strip().casefold()')
    edit(9, '    if not isinstance(specs, list) or not specs:',
         '    if not isinstance(specs, list) or not specs or any(not isinstance(spec, dict) for spec in specs):')
    edit(9, '    used = {spec.get("file_index") for spec in specs}',
         '    indices = [spec.get("file_index") for spec in specs]\n    if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):\n        raise ValueError("file_index must be an integer")\n    if len(ignored) != len(set(ignored)):\n        raise ValueError("ignored_file_indices must be distinct")\n    used = set(indices)')
    edit(9, 'Every input file must be used or explicitly ignored exactly once',
         'Every input file must be used in one or more views, or explicitly ignored')
    edit(9, '            raw_values = condition.get("value", [])\n            raw_values = raw_values if isinstance(raw_values, list) else [raw_values]\n            value_supported = bool(raw_values) and all(key(value) in key(query) for value in raw_values)\n            if not evidence.strip() or (key(evidence) not in key(query) and not value_supported):',
         '            if not evidence.strip() or key(evidence) not in key(query):')
    edit(9, '        row_axis_id, column_axis_id = row_axis_spec.get("table_id"), column_axis_spec.get("table_id")',
         '        if not isinstance(row_axis_spec, dict) or not isinstance(column_axis_spec, dict):\n            raise ValueError("Matrix axis specifications must be objects")\n        row_axis_id, column_axis_id = row_axis_spec.get("table_id"), column_axis_spec.get("table_id")')
    edit(9, '"row_ids_aligned": axis_keys(matrix_row_ids, "matrix row") == axis_keys(row_ids, "row axis"),\n            "column_ids_aligned": axis_keys(matrix_columns, "matrix column") == axis_keys(column_ids, "column axis"),',
         '"row_ids_aligned": set(axis_keys(matrix_row_ids, "matrix row")) == set(axis_keys(row_ids, "row axis")),\n            "column_ids_aligned": set(axis_keys(matrix_columns, "matrix column")) == set(axis_keys(column_ids, "column axis")),\n            "row_order_matches": matrix_row_ids == row_ids,\n            "column_order_matches": matrix_columns == column_ids,')
    edit(15, 'values or literal record counts."""',
         'values or literal record counts. When CSVQA has applied a validated filter, use its returned\nrecords directly and retain that exact predicate in Data Mapping; never re-filter by a guessed\ncomplete label. For FALLBACK_FULL_DATA, state and implement the query-supported selection explicitly."""')
    edit(25, 'Create exactly one Gurobi model in m or model and call its optimize() exactly once, both at top level.',
         'Create exactly one Gurobi model in m and call its optimize() exactly once, both at top level.')
    edit(25, 'stale variables from a preceding loop. Do not silently skip failed conversions or missing coefficients.',
         'stale variables from a preceding loop. Do not silently skip failed conversions or missing coefficients.\nUse Series.str.strip().str.casefold() for vectorized string normalization, or apply a scalar function;\nSeries has no casefold() method. For arbitrary CSV headers (spaces, punctuation, numeric names),\nuse row[column] or explicit arrays, not getattr on itertuples(), which renames invalid field names.')
    edit(25, 'A function that creates a model must explicitly return m, and its caller must assign that result to m.',
         'Create and solve m at top level as required above; keep it live until result extraction.')
    # Prefix templates require exactly one escape, while input values require none.
    edit(25, '        label = label.replace("{", "{{").replace("}", "}}")\n', '')
    edit(25, ' (If not mentioned or ambiguous, integer by default!)',
         ' (Use integers for indivisible item counts, binaries for choices, and continuous variables for explicitly divisible flows or material.)')
    edit(25, "    query = query.replace('{','{{').replace('}','}}')\n", '')
    edit(27, '    New runs use planned JSON payloads. This compatibility parser accepts JSON',
         '    Legacy runs use source-tagged JSON rows. This compatibility parser accepts JSON')
    edit(27, '{str(key).strip(): value for key, value in values.items()}',
         '{str(key): value for key, value in values.items()}')
    edit(27, 'if not headers or len(headers) != len(separator) or any(not h for h in headers):',
         'if not headers or len(headers) != len(separator) or len(headers) != len(set(headers)) or any(not h for h in headers):')
    edit(27, 'if not reader.fieldnames or any(field is None or not field.strip() for field in reader.fieldnames):',
         'if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or any(field is None or not field.strip() for field in reader.fieldnames):')
    edit(27, 'Leave the solved Gurobi model in m or model. If using a function, return the model',
         'Leave the solved Gurobi model in m. If using a function, return the model')
    edit(27, 'Read source CSVs with dtype=str and keep_default_na=False, then explicitly convert the required numeric fields.',
         'When the task permits CSV reading, use dtype=str and keep_default_na=False, then explicitly convert the required numeric fields; otherwise use only the supplied records or formulation data.')
    edit(31, '    frame["problem_id"] = frame["problem_id"].astype(str)',
         '    if frame.empty or frame["problem_id"].isna().any():\n        raise ValueError("Cases require nonempty IDs")\n    frame["problem_id"] = frame["problem_id"].astype(str)\n    if frame["problem_id"].str.strip().eq("").any() or frame["problem_id"].duplicated().any():\n        raise ValueError("Cases require unique nonempty IDs")')
    edit(33, '("final_ok", "classification_correct", "solution_correct")',
         '("final_ok", "classification_correct", "solution_correct", "external_deadline_exceeded", "route_allowed")')
    edit(33, '        artifact = root / relative if relative else None\n        if artifact is None',
         '        if not relative and not expected_hash and row.get("record_status") == "error":\n            continue  # A failed stage may never have produced this artifact.\n        artifact = root / relative if relative else None\n        if artifact is None')
    edit(33, '"""Reuse only successful, intact results from the same code and data."""\n    if (not record or record.get("record_status") != "completed"\n            or not record.get("final_ok") or record.get("cache_fingerprint") != fingerprint):',
         '"""Resume intact recorded attempts, including failures, without selective retries."""\n    if (not record or record.get("record_status") not in {"completed", "error"}\n            or record.get("cache_fingerprint") != fingerprint):')
    edit(35, '"""Reuse matching successful results; retry failures. False disables all result reuse."""',
         '"""Resume the same immutable round; use a fresh output directory for a new round."""')
    edit(35, '    cached = {record_key(row): row for row in load_records(output_csv)} if reuse_completed else {}',
         '    existing = load_records(output_csv)\n    if existing and not reuse_completed:\n        raise ValueError("Use a fresh round directory; existing attempts cannot be overwritten")\n    cached = {record_key(row): row for row in existing}')
    edit(35, '            record = reusable_record(output_csv, cached.get(key), fingerprint)',
         '            previous = cached.get(key)\n            if previous and previous.get("cache_fingerprint") != fingerprint:\n                raise ValueError("Code/data changed; use a fresh version directory")\n            record = reusable_record(output_csv, previous, fingerprint)\n            if previous and record is None:\n                raise ValueError("Recorded artifacts are missing/corrupt; do not replace the attempt")')
    edit(3, 'outputs/optimization_0927_20261006/full_v1', 'outputs/optimization_0927_20261006/full_review_v2')
    edit(3, 'NOTEBOOK_FILENAME = "LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb"',
         f'NOTEBOOK_FILENAME = {str(DEST)!r}')
    edit(3, 'EXPERIMENT_METHOD = "examples_and_route"', 'EXPERIMENT_METHOD = "full"')
    # The full pipeline has no classification cache; avoid a stale cache dependency.
    config = "".join(book["cells"][3]["source"])
    config = "\n".join(line for line in config.splitlines() if not line.startswith(("BASE_NOTEBOOK_", "CLASSIFICATION_RESULTS_"))) + "\n"
    book["cells"][3]["source"] = config.splitlines(True)
    book["metadata"]["review_candidate"] = {"baseline": str(BASE), "baseline_sha256": sha(BASE), "revision": "review-v2"}
    for i, cell in enumerate(book["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
            cell["outputs"] = []
            cell["execution_count"] = None
    DEST.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
    if DEST.exists() and DEST.read_text() != serialized:
        raise ValueError("Candidate already exists with different source; create a new revision")
    DEST.write_text(serialized)
    nbformat.validate(nbformat.read(DEST, as_version=4))
    (OUT / "patch_v2.json").write_text(json.dumps(changed, ensure_ascii=False, indent=2))
    print(json.dumps({"candidate": str(DEST), "sha256": sha(DEST), "changed_cells": sorted({r["cell"] for r in changed}), "changes": len(changed)}))


if __name__ == "__main__":
    main()
