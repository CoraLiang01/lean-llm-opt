"""Fix demonstrated formatting compatibility defects; preserve the v2 attempt."""
import json
import hashlib
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
SOURCE = OUT / "candidates/full_review_v2.ipynb"
DEST = OUT / "candidates/full_review_v3.ipynb"


def main():
    book = json.loads(SOURCE.read_text())
    edits = []

    def edit(index, old, new):
        source = "".join(book["cells"][index]["source"])
        assert source.count(old) == 1, (index, old[:80])
        book["cells"][index]["source"] = source.replace(old, new).splitlines(True)
        edits.append({"cell": index, "old": old, "new": new})

    edit(9, 'when the query describes a family and the profile shows no exact match. Preserve literal identifiers.',
         'when the query describes a family and the profile shows no exact match. For family/category codes,\nprefer an exact match when present, otherwise a supported prefix if present; use unrestricted\ncontains only if the query asks for an anywhere-substring or no exact/prefix match exists.\nScalar predicates (exact, eq, ne, prefix, contains, gt, ge, lt, le) require a scalar value, not an array.\nCopy evidence from the query; typographic quotation marks may be preserved. Preserve literal identifiers.')
    edit(9, '    if isinstance(value, list) and operator not in {"in", "not_in", "between"}:\n        raise ValueError("Scalar operators require a scalar value")',
         '    if isinstance(value, list) and operator not in {"in", "not_in", "between"}:\n        if len(value) != 1:\n            raise ValueError("Scalar operators require a scalar value")\n        value = value[0]  # A singleton has one unambiguous scalar meaning.')
    edit(9, '    key = lambda value: re.sub(r"\\s+", " ", str(value)).strip().casefold()',
         '    quote_map = str.maketrans({"“": \'"\', "”": \'"\', "‘": "\'", "’": "\'"})\n    key = lambda value: re.sub(r"\\s+", " ", str(value).translate(quote_map)).strip().casefold()')
    edit(9, '            normalized = [key(value) for value in values]',
         '            normalized = [str(value).strip() for value in values]')
    edit(15, '    frame = read_csv_compat(RAG_EXAMPLES_ALL_PATH, dtype=str, keep_default_na=False)',
         '    reference_path = Path(RAG_EXAMPLES_ALL_PATH).expanduser()\n    reference_path = reference_path if reference_path.is_absolute() else PROJECT_ROOT / reference_path\n    frame = read_csv_compat(reference_path, dtype=str, keep_default_na=False)')
    edit(15, '            frame = read_csv_compat(raw.strip(), dtype=str, keep_default_na=False)',
         '            path = Path(raw.strip()).expanduser()\n            path = path if path.is_absolute() else PROJECT_ROOT / path\n            frame = read_csv_compat(path, dtype=str, keep_default_na=False)')
    edit(25, 'use row[column] or explicit arrays, not getattr on itertuples(), which renames invalid field names.',
         'use row[column] or explicit arrays, not getattr on itertuples(), which renames invalid field names.\nSeries.to_dict() uses the DataFrame index, not an ID column: set the validated entity ID as index\nor construct dictionaries with explicit ID/value pairs. Preserve identifier case by default. If\nnormalization is required, apply the same mapping to entity values and matrix column labels.\nWhen a supplied identity/lookup table relates opaque references to entity IDs, resolve foreign\nkeys through that table and validate coverage; never compare an opaque reference directly to\na display label as if they were the same key.')
    edit(29, '    except Exception as exc:\n        record.update(api_retry_count=',
         '    except BaseException as exc:\n        record.update(api_retry_count=')
    edit(29, '        record.update(predicted_label=label, assigned_route=route if address else "Others")',
         '        route = route if address else "Others"\n        record.update(predicted_label=label, assigned_route=route)')
    edit(3, 'outputs/optimization_0927_20261006/full_review_v2', 'outputs/optimization_0927_20261006/full_review_v3')
    config = "".join(book["cells"][3]["source"])
    config = config.replace(str(SOURCE), str(DEST))
    book["cells"][3]["source"] = config.splitlines(True)
    book["metadata"]["review_candidate"]["revision"] = "review-v3"
    book["metadata"]["review_candidate"]["compatibility_fixes"] = "Quote typography; singleton scalar predicates; family-code matching; case-sensitive matrix IDs; interrupted-stage context. No case IDs or reference answers."
    for i, c in enumerate(book["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    text = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
    if DEST.exists() and DEST.read_text() != text:
        if (ROOT / "outputs/optimization_0927_20261006/full_review_v3/frozen_notebook.ipynb").exists():
            raise ValueError("Frozen candidate cannot change; use a new revision")
        digest = hashlib.sha256(DEST.read_bytes()).hexdigest()[:12]
        draft = OUT / f"candidate_drafts/full_review_v3_draft_{digest}.ipynb"
        draft.parent.mkdir(parents=True, exist_ok=True)
        if not draft.exists():
            draft.write_bytes(DEST.read_bytes())
    DEST.write_text(text)
    nbformat.validate(nbformat.read(DEST, as_version=4))
    (OUT / "patch_v3.json").write_text(json.dumps(edits, ensure_ascii=False, indent=2))
    print(DEST)


if __name__ == "__main__":
    main()
