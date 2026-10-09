"""Apply minimal fragment completeness validation to the preserved v10 sources."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


FRAGMENT_CHECK = '''        # Same-schema files can be fragments of a required logical table.
        for ignored_index in ignored:
            omitted = by_index[ignored_index]["frame"]
            if set(omitted.columns) != set(frame.columns):
                continue
            omitted_mask = None
            for condition in conditions:
                mask = _apply_condition(omitted[condition["column"]], condition)
                omitted_mask = mask if omitted_mask is None else (
                    omitted_mask & mask if logic == "and" else omitted_mask | mask)
            candidates = omitted if omitted_mask is None else omitted.loc[omitted_mask]
            if "table" in frame.columns:
                candidates = candidates.loc[candidates["table"].isin(selected["table"])]
            if not candidates.empty:
                raise ValueError(
                    f"Ignored CSV fragment {ignored_index} contains query-matching rows "
                    f"for required file {file_index}; preserve all source fragments")
'''

PLANNER_GUIDANCE = '''Files with the same schema may be fragments of one required logical table. Include every
required fragment; profile samples cannot justify dropping a file. Apply explicit query filters
to the full source, then combine logical-table fragments before selecting latest versions.
'''


def main():
    backup = Path(json.loads((OUT / 'v11_backup_location.json').read_text())['backup'])
    books = {method: json.loads((backup / name).read_text()) for method, name in NAMES.items()}
    changes, modifications = {}, {}
    for method, book in books.items():
        changed, diff = [], []
        for index, cell in enumerate(book['cells']):
            before = ''.join(cell['source'])
            source = before.replace('_v10', '_v11').replace('react-source-data-v10', 'react-source-data-v11')
            if cell['cell_type'] == 'code' and 'def _execute_plan(' in source:
                marker = '        records = [\n'
                assert source.count(marker) == 1
                source = source.replace(marker, FRAGMENT_CHECK + marker, 1)
                marker = 'Filter only an explicit restriction;'
                assert source.count(marker) == 1
                source = source.replace(marker, PLANNER_GUIDANCE + marker, 1)
            if cell['cell_type'] == 'code' and 'MODEL_CONSISTENCY_INSTRUCTIONS =' in source:
                tree = ast.parse(source)
                node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'MODEL_CONSISTENCY_INSTRUCTIONS'
                    for target in n.targets))
                guidance = ast.literal_eval(node.value)
                original = 'Use single ordinary ASCII spaces and newlines for formatting. Do not align columns,\npad lines with Unicode spaces, or use tables; preserve exact source identifiers.\n'
                assert original in guidance
                guidance = guidance.replace(original,
                    'Write a concise Markdown model with short lines and $...$ equations; preserve exact source identifiers.\n')
                source = source.replace(ast.get_source_segment(source, node),
                    'MODEL_CONSISTENCY_INSTRUCTIONS = ' + repr(guidance), 1)
            if cell['cell_type'] == 'code':
                ast.parse(source)
            if source != before:
                cell['source'] = source.splitlines(keepends=True)
                if cell['cell_type'] == 'code':
                    cell['execution_count'], cell['outputs'] = None, []
                changed.append(index)
                # The configuration diff includes only version identifiers, never credentials.
                left = [line for line in before.splitlines(True) if not re.search(r'API_KEY|api_key', line)]
                right = [line for line in source.splitlines(True) if not re.search(r'API_KEY|api_key', line)]
                diff.extend(difflib.unified_diff(left, right,
                    fromfile=f'{NAMES[method]}:v10:cell{index}', tofile=f'{NAMES[method]}:v11:cell{index}'))
        changes[method], modifications[method] = changed, ''.join(diff)
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
    for method, diff in modifications.items():
        (OUT / f'{Path(NAMES[method]).stem}_v10_to_v11.diff').write_text(diff)
    scope = {'version': 'v11', 'source': 'preserved v10', 'backup': str(backup),
             'full_sha256': full_sha, 'changed_cells': changes,
             'behavioral_changes': ['Validate ignored same-schema fragments against selected query filters; use existing full-source fallback',
                                    'Explain fragment completeness to the extraction planner',
                                    'Replace ineffective negative formatting instruction with concise positive Markdown guidance'],
             'react_preserved': True, 'repair_added': False, 'outcome_retry_added': False,
             'source_data_or_tolerances_changed': False, 'query_only_formulation_changed': False,
             'performance_status': 'unverified until independent complete full pass', '606_enabled': False}
    (OUT / 'revision_scope_v11.json').write_text(json.dumps(scope, indent=2))
    print(json.dumps(scope, indent=2))


if __name__ == '__main__':
    main()
