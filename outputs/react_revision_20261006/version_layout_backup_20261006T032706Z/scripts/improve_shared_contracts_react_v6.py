"""Minimal shared v6 guidance from preserved v5 failures; no case-specific generation."""
import ast
import hashlib
import json
from pathlib import Path
import re

from restore_react_revision_20261006 import ROOT, OUT, FILES, save

SEMANTICS = (
    "Apply unconditional quantity/resource bounds unconditionally. Only use activation-conditioned "
    "bounds when the current query explicitly makes them conditional; an activation fee alone "
    "does not make an otherwise unconditional bound conditional."
)


def main():
    books = {}
    for method, name in FILES.items():
        book = json.loads((OUT / 'notebook_snapshots/v5' / name).read_text())
        for cell in book['cells']:
            source = ''.join(cell['source'])
            source = source.replace('_v5', '_v6').replace('react-source-data-v5', 'react-source-data-v6')
            source = source.replace(
                'Keep decision-variable containers distinct from loop, record and parameter names; never overwrite them.',
                "Give every decision-variable container a distinct name ending in _vars, such as quantity_vars. "
                "Keep scalar keys, loop variables, row objects and parameters separate; never rebind a decision container.\n"
                "quicksum requires an iterable of numeric or linear terms; use scalar constants directly, never quicksum(0).\n"
                "Read external CSVs with dtype=str and keep_default_na=False, then explicitly convert required numeric fields. "
                "Preserve empty fields and literal identifiers; do not turn a missing relationship into a string 'nan'.\n" + SEMANTICS)
            if 'def formulate_with_csvqa(' in source:
                marker = '    suffix = """Begin!'
                assert marker in source
                source = source.replace(marker, '    prefix += ' + repr('\n' + SEMANTICS) + '\n' + marker, 1)
            if 'def get_Others_response(' in source:
                marker = '        abstract_prompt = PromptTemplate('
                assert marker in source
                source = source.replace(marker,
                    '        abstract_model_template += ' + repr('\n' + SEMANTICS) + '\n\n' + marker, 1)
            cell['source'] = source.splitlines(keepends=True)
        books[method] = book
    save(ROOT / FILES['full'], books['full'])
    full_sha = hashlib.sha256((ROOT / FILES['full']).read_bytes()).hexdigest()
    for method, book in books.items():
        if method == 'full':
            continue
        for cell in book['cells']:
            source = ''.join(cell['source'])
            if method in {'rag_only', 'few_shot_only'}:
                source = re.sub(r'BASE_NOTEBOOK_SHA256 = "[a-f0-9]+"',
                                f'BASE_NOTEBOOK_SHA256 = "{full_sha}"', source)
            else:
                source = re.sub(r'LOTO_BASE_SHA256 = "[a-f0-9]+"',
                                f'LOTO_BASE_SHA256 = "{full_sha}"', source)
            cell['source'] = source.splitlines(keepends=True)
        save(ROOT / FILES[method], book)
    (OUT / 'revision_scope_v6.json').write_text(json.dumps({
        'version': 'v6', 'files': FILES, 'architecture': 'Original ReAct; v5 user-authorized protocol policy retained',
        'shared_guidance': ['distinct _vars decision containers', 'iterable quicksum / direct scalar constants',
                            'raw CSV strings and empty fields', 'unconditional bounds remain unconditional'],
        'repair': False, 'objective_based_retry': False, 'csvqa_minimum_calls': 1,
        'multiple_csvqa_calls': True, 'protocol_restarts': True, '606_enabled': False,
        'previous_results': 'All v5 attempts retained; no case-wise result selection',
    }, indent=2))
    print('Built v6 with shared data/API/semantic guidance and unchanged ReAct retry policy.')


if __name__ == '__main__':
    main()
