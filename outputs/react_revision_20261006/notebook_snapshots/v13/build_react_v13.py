"""Fix a lossless filter-value interface mismatch in the preserved v12 sources."""
import ast
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


VALUE_ADAPTER = '''    if isinstance(value, list) and operator not in {"in", "not_in", "between"}:
        if len(value) != 1:
            raise ValueError(f"Operator {operator!r} requires one scalar value")
        value = value[0]
'''


def main():
    backup = Path(json.loads((OUT / 'v13_backup_location.json').read_text())['backup'])
    books = {method: json.loads((backup / name).read_text()) for method, name in NAMES.items()}
    changes = {}
    for method, book in books.items():
        changed = []
        for index, cell in enumerate(book['cells']):
            before = ''.join(cell['source'])
            source = before.replace('_v12', '_v13').replace('react-source-data-v12', 'react-source-data-v13')
            if cell['cell_type'] == 'code' and 'def _apply_condition(' in source:
                marker = '    value = condition.get("value")\n'
                assert source.count(marker) == 1
                source = source.replace(marker, marker + VALUE_ADAPTER, 1)
                marker = 'Choose the filter operator from the requested matching semantics.'
                assert source.count(marker) == 1
                source = source.replace(marker,
                    'Use scalar values for scalar operators; use lists only for in, not_in and between.\n'
                    'For multiple prefixes, declare separate conditions with the requested and/or logic.\n' + marker, 1)
            if cell['cell_type'] == 'code' and 'MODEL_CONSISTENCY_INSTRUCTIONS =' in source:
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'MODEL_CONSISTENCY_INSTRUCTIONS' for t in n.targets))
                guidance = ast.literal_eval(node.value)
                marker = 'Write a concise Markdown model with short lines and $...$ equations; preserve exact source identifiers.\n'
                assert guidance.count(marker) == 1
                # Preserve the original ReAct example/model format; extra format wording has no verified benefit.
                guidance = guidance.replace(marker, '')
                source = source.replace(ast.get_source_segment(source, node),
                    'MODEL_CONSISTENCY_INSTRUCTIONS = ' + repr(guidance), 1)
            if cell['cell_type'] == 'code':
                ast.parse(source)
            if source != before:
                cell['source'] = source.splitlines(keepends=True)
                if cell['cell_type'] == 'code':
                    cell['execution_count'], cell['outputs'] = None, []
                changed.append(index)
        changes[method] = changed
    full = ROOT / NAMES['full']
    full.write_text(json.dumps(books['full'], ensure_ascii=False, indent=1) + '\n')
    full_sha = hashlib.sha256(full.read_bytes()).hexdigest()
    for method, book in books.items():
        if method == 'full':
            continue
        for cell in book['cells']:
            source = re.sub(r'((?:BASE_NOTEBOOK_SHA256|LOTO_BASE_SHA256) = )"[a-f0-9]+"',
                            lambda match: match.group(1) + '"' + full_sha + '"', ''.join(cell['source']))
            cell['source'] = source.splitlines(keepends=True)
        (ROOT / NAMES[method]).write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')
    scope = {'version': 'v13', 'source': 'preserved v12', 'backup': str(backup),
             'full_sha256': full_sha, 'changed_cells': changes,
             'behavioral_changes': ['Lossless singleton-list value adaptation for scalar filter operators; reject ambiguous multi-value forms',
                                    'Clarify planner scalar/list operator contracts',
                                    'Remove extra unverified formatting instruction; retain original ReAct example/model format'],
             'react_preserved': True, 'repair_added': False, 'outcome_retry_added': False,
             'source_data_or_tolerances_changed': False, 'query_only_formulation_changed': False,
             'performance_status': 'unverified until complete independent pass', '606_enabled': False}
    (OUT / 'revision_scope_v13.json').write_text(json.dumps(scope, indent=2))
    print(json.dumps(scope, indent=2))


if __name__ == '__main__':
    main()
