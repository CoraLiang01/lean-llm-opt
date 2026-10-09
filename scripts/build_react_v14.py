"""Remove one conflicting shared prompt statement from the preserved v13 family."""
import ast
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


def main():
    backup = Path(json.loads((OUT / "v14_backup_location.json").read_text())["backup"])
    books = {method: json.loads((backup / name).read_text()) for method, name in NAMES.items()}
    changed = {}
    for method, book in books.items():
        cells, removed = [], 0
        for index, cell in enumerate(book["cells"]):
            before = "".join(cell["source"])
            source = before.replace("_v13", "_v14").replace("react-source-data-v13", "react-source-data-v14")
            lines = source.splitlines(keepends=True)
            deletion = [line for line in lines if "prefix += " in line and
                        "Only use activation-conditioned bounds when the current query explicitly" in line]
            removed += len(deletion)
            source = "".join(line for line in lines if line not in deletion)
            if cell["cell_type"] == "code":
                ast.parse(source)
            if source != before:
                cell["source"] = source.splitlines(keepends=True)
                if cell["cell_type"] == "code":
                    cell["execution_count"], cell["outputs"] = None, []
                cells.append(index)
        assert removed == 1, (method, removed)
        changed[method] = cells
    full = ROOT / NAMES["full"]
    full.write_text(json.dumps(books["full"], ensure_ascii=False, indent=1) + "\n")
    full_sha = hashlib.sha256(full.read_bytes()).hexdigest()
    for method, book in books.items():
        if method == "full":
            continue
        for cell in book["cells"]:
            source = re.sub(r'((?:BASE_NOTEBOOK_SHA256|LOTO_BASE_SHA256) = )"[a-f0-9]+"',
                            lambda match: match.group(1) + '"' + full_sha + '"', "".join(cell["source"]))
            cell["source"] = source.splitlines(keepends=True)
        (ROOT / NAMES[method]).write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n")
    scope = {"version": "v14", "source": "preserved v13", "backup": str(backup),
             "full_sha256": full_sha, "changed_cells": changed,
             "behavioral_change": "Delete one overrestrictive activation-bound prefix statement that conflicts with required operational links; existing general guidance remains.",
             "react_preserved": True, "repair_added": False, "outcome_retry_added": False,
             "classification_or_source_data_changed": False, "tolerances_changed": False,
             "query_only_formulation_changed": False, "606_enabled": False,
             "performance_status": "Unverified; API credit_balance_exhausted prevents a new evaluation",
             "inference_cases": 0}
    (OUT / "revision_scope_v14.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    main()
