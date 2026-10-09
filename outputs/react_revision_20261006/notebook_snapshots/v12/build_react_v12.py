"""Add one shared source-identifier parsing instruction to preserved v11 code."""
import ast
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


IDENTIFIER_GUIDANCE = '''When extracting coefficients from free text, match each complete source identifier.
Escape identifiers used in regular expressions and prevent a shorter identifier from
matching a substring of another identifier.
'''


def main():
    backup = Path(json.loads((OUT / 'v12_backup_location.json').read_text())['backup'])
    books = {method: json.loads((backup / name).read_text()) for method, name in NAMES.items()}
    changes = {}
    for method, book in books.items():
        changed = []
        for index, cell in enumerate(book['cells']):
            before = ''.join(cell['source'])
            source = before.replace('_v11', '_v12').replace('react-source-data-v11', 'react-source-data-v12')
            if cell['cell_type'] == 'code' and 'CSV_SOLVER_INSTRUCTIONS =' in source:
                marker = 'Set MIPGap=1e-4 before optimize().'
                assert source.count(marker) == 1
                source = source.replace(marker, IDENTIFIER_GUIDANCE + marker, 1)
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
    scope = {'version': 'v12', 'source': 'preserved v11 preflight-only source', 'backup': str(backup),
             'full_sha256': full_sha, 'changed_cells': changes,
             'behavioral_changes': 'One shared whole-identifier parsing instruction, supported by the saved v10 source/parser collision',
             'v11_inference_cases': 0, 'react_preserved': True, 'repair_added': False,
             'outcome_retry_added': False, 'source_data_or_tolerances_changed': False,
             'query_only_formulation_changed': False, 'performance_status': 'unverified until complete independent pass',
             '606_enabled': False}
    (OUT / 'revision_scope_v12.json').write_text(json.dumps(scope, indent=2))
    print(json.dumps(scope, indent=2))


if __name__ == '__main__':
    main()
