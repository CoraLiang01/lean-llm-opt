"""Preserve the evaluated direct-call delivery and relocate ReAct without model edits."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/react_revision_20261006"
SOURCE = {
    "full": "LEAN_LLM_OPT_4.1_Large-scale_1006.ipynb",
    "rag_only": "Ablation_Study_Large_Scale_Or_RAG_Only.ipynb",
    "few_shot_only": "Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb",
    "examples_only": "LOTO_Examples_Only_GPT4.1_Large-scale.ipynb",
    "examples_and_route": "LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb",
}
DIRECT = {**SOURCE, "full": "LEAN_LLM_OPT_4.1_Large-scale.ipynb"}
REACT = {method: name.removesuffix(".ipynb") + "_ReAct.ipynb"
         for method, name in SOURCE.items()}
RUNS = {"full": "full_v4", "rag_only": "rag_only_v4",
        "few_shot_only": "few_shot_only_v5", "examples_only": "examples_only_v4",
        "examples_and_route": "examples_and_route_v4"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(book):
    cells = []
    for cell in book["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code":
            text = re.sub(r'(?m)^(NOTEBOOK_FILENAME|LOTO_BASE_NOTEBOOK) = "[^"\n]+"$',
                          lambda m: m.group(1) + ' = "<relocated_filename>"', text)
        cells.append((cell["cell_type"], text))
    return cells


def main():
    backup = Path(json.loads((OUT / "version_layout_backup_pointer.json").read_text())["backup_directory"])
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "pre_change_backup": str(backup), "changes": "Filename/reference configuration only",
             "inference_function_changes": 0, "historical_results_modified": False, "methods": {}}
    for method, old_name in SOURCE.items():
        before = (backup / old_name).read_bytes()
        book = json.loads(before)
        changes = []
        for i, cell in enumerate(book["cells"]):
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            changed = re.sub(r'(?m)^NOTEBOOK_FILENAME = "[^"\n]+"$',
                             f'NOTEBOOK_FILENAME = "{REACT[method]}"', text)
            changed = re.sub(r'(?m)^LOTO_BASE_NOTEBOOK = "[^"\n]+"$',
                             f'LOTO_BASE_NOTEBOOK = "{REACT["full"]}"', changed)
            if changed != text:
                changes.append(i)
                cell["source"] = changed.splitlines(keepends=True)
            ast.parse(changed)
        assert normalized(json.loads(before)) == normalized(book)
        after = (json.dumps(book, indent=1, ensure_ascii=False) + "\n").encode()
        target = ROOT / REACT[method]
        if target.exists():
            assert target.read_bytes() == after, f"Refusing to overwrite {target}"
        target.write_bytes(after)

        direct = (OUT / "backups_direct_call_version" / old_name).read_bytes()
        direct_target = ROOT / DIRECT[method]
        if direct_target.exists():
            assert direct_target.read_bytes() in (before, direct), f"Unexpected existing version: {direct_target}"
        direct_target.write_bytes(direct)
        frozen = ROOT / "outputs/minimal_revision_20261005" / RUNS[method] / "frozen_notebook.ipynb"
        evaluated = json.loads(frozen.read_text())
        delivered = json.loads(direct)
        changed_code = [i for i, (a, b) in enumerate(zip(evaluated["cells"], delivered["cells"]))
                        if a["cell_type"] == "code" and a["source"] != b["source"]]
        assert changed_code == ({"full": [39], "examples_only": [39],
                                 "examples_and_route": [39]}.get(method, []))
        for cell in delivered["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        audit["methods"][method] = {
            "direct_call_notebook": DIRECT[method], "react_notebook": REACT[method],
            "direct_delivery_sha256": sha(direct), "direct_evaluated_run": RUNS[method],
            "direct_evaluated_sha256": sha(frozen.read_bytes()),
            "direct_post_evaluation_changed_code_cells": changed_code,
            "direct_restored_byte_for_byte": direct_target.read_bytes() == direct,
            "react_before_rename_sha256": sha(before), "react_after_rename_sha256": sha(after),
            "react_changed_cells": changes, "react_only_filename_assignments_changed": True,
        }
    previous_full = ROOT / SOURCE["full"]
    if previous_full.exists():
        assert previous_full.read_bytes() == (backup / SOURCE["full"]).read_bytes()
        previous_full.unlink()
    (OUT / "version_layout_20261006.json").write_text(json.dumps(audit, indent=2))
    lines = ["# Preserved notebook families", "",
             "The direct-call delivery is restored byte for byte from its pre-ReAct backup.",
             "Its scores belong to outputs/minimal_revision_20261005. ReAct evaluation records",
             "remain under outputs/react_revision_20261006. No results are combined across families.", "",
             "| Method | Preserved direct-call notebook | Current ReAct notebook |",
             "| --- | --- | --- |"]
    for method in SOURCE:
        lines.append(f"| {method} | [{DIRECT[method]}]({ROOT / DIRECT[method]}) | [{REACT[method]}]({ROOT / REACT[method]}) |")
    lines += ["", "Only NOTEBOOK_FILENAME and the two LOTO_BASE_NOTEBOOK filename assignments",
              "changed in the ReAct notebooks during this separation. Modeling prompts, ReAct",
              "agents, CSVQA, execution, tolerances and solver settings were not changed.",
              "The classification-cache baseline hashes continue to identify the actual frozen",
              "full_v7 evaluation, not a different model. The evaluator permits self filename",
              "relocation while requiring all other definition cells to remain identical.", "",
              "The original direct-call full model has planned canonical CSV routes and the",
              "original legacy Others CSV flow. Few-shot Only deliberately overrides CSV modes",
              "to direct full-source Observation. Classification and query-only Others keep their",
              "original architectures. No mode was changed simply to relabel the preserved code.", "",
              "The restored direct full notebook retains its previously enabled 606 switch;",
              "606 was not executed. The ReAct 606 switch remains disabled pending its own gate.",
              "Both result sets and all frozen notebook snapshots remain intact.", "",
              "The full delivery differs from its evaluated snapshot only in the reviewed 606",
              "control cell. The two direct LOTO deliveries retain the documented no-CSV alias",
              "correction; that branch was not API-evaluated by the all-CSV campaign.", ""]
    (OUT / "NOTEBOOK_FAMILIES.md").write_text("\n".join(lines))
    print(json.dumps({"direct": DIRECT, "react": REACT, "inference_function_changes": 0}, indent=2))


if __name__ == "__main__":
    main()
