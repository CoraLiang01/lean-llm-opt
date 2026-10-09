"""Apply small shared prompt corrections to the preserved v9 ReAct sources."""
import ast
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


MODEL_CONSISTENCY = """
Use single ordinary ASCII spaces and newlines for formatting. Do not align columns,
pad lines with Unicode spaces, or use tables; preserve exact source identifiers.
Before the Final Answer, check units, conservation signs and bound applicability.
Derive interval durations from source time labels; a row is not necessarily one hour.
For a closed flow system use one consistent sign convention and balance totals of zero.
Unconditional quantity limits must not be multiplied by activation variables.
Retain activation links required for operating/open decisions: such logical links are
separate from unconditional limits and may use valid bounds derived from current data.
"""

GUROBI_API = """
Omit ub rather than writing ub=None; for example, addVars(keys, vtype=gp.GRB.INTEGER,
name='') has nonnegative variables with no finite upper bound. Never pass None as a bound.
gp.min_, max_, abs_, and_, or_ are general expressions, not linear terms: equate them
to a single auxiliary variable; never place them in an inequality or objective sum.
An indicator trigger must be a separately defined GRB.BINARY variable, never an integer
quantity or a continuous variable. Link quantity to binary activation with valid bounds
derived from the data, preserving unconditional limits and the original variable domain.
Combine fragments of one logical table before applying latest-revision, deletion or
deduplication rules. Use the complete query-defined grouping key and selection order,
then aggregate; selecting versions separately per file can retain superseded records.
"""


def main():
    backup = Path(json.loads((OUT / 'v10_backup_location.json').read_text())['backup'])
    changes = {}
    books = {method: json.loads((backup / name).read_text()) for method, name in NAMES.items()}
    for method, book in books.items():
        changed = []
        for index, cell in enumerate(book['cells']):
            before = ''.join(cell['source'])
            source = before.replace('_v9', '_v10').replace('react-source-data-v9', 'react-source-data-v10')
            if cell['cell_type'] == 'code' and 'def formulate_with_csvqa(' in source:
                node = next(n for n in ast.parse(source).body
                            if isinstance(n, ast.FunctionDef) and n.name == 'formulate_with_csvqa')
                original_function = ast.get_source_segment(source, node)
                marker = '{agent_scratchpad}"""'
                assert original_function.count(marker) == 1
                updated_function = original_function.replace(marker, marker + '\n    suffix = suffix.replace("{agent_scratchpad}", MODEL_CONSISTENCY_INSTRUCTIONS + "\\n{agent_scratchpad}")')
                source = ('MODEL_CONSISTENCY_INSTRUCTIONS = ' + repr(MODEL_CONSISTENCY) + '\n\n'
                          + source.replace(original_function, updated_function, 1))
            if cell['cell_type'] == 'code' and 'CSV_SOLVER_INSTRUCTIONS =' in source:
                source = source.replace('Set MIPGap=1e-4 before optimize().', GUROBI_API + '\nSet MIPGap=1e-4 before optimize().')
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
    scope = {'version': 'v10', 'source': 'preserved v9', 'backup': str(backup),
             'full_sha256': full_sha, 'changed_cells': changes,
             'behavioral_changes': 'Shared formulation formatting/consistency and Gurobi API prompts only',
             'react_preserved': True, 'repair_added': False, 'outcome_retry_added': False,
             'source_data_or_tolerances_changed': False, 'query_only_formulation_changed': False,
             'performance_status': 'unverified until independent complete full pass', '606_enabled': False}
    (OUT / 'revision_scope_v10.json').write_text(json.dumps(scope, indent=2))
    print(json.dumps(scope, indent=2))


if __name__ == '__main__':
    main()
