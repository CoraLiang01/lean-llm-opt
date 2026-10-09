"""Apply evidence-based shared corrections to the backed-up ReAct family only."""
import ast
import hashlib
import json
from pathlib import Path
import re

from evaluate_react_revision_20261006 import ROOT, OUT, NAMES


def replace_function(source, name, replacement):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


SOURCE_CANDIDATE = '''def _source_candidate(code):
    """Remove a complete Markdown fence and validate syntax without changing program semantics."""
    source = str(code).strip()
    match = re.fullmatch(r"```(?:python|py)?\\s*\\n(.*?)\\n```", source, re.DOTALL)
    source = match.group(1) if match else source
    ast.parse(source)
    return source
'''

OTHERS = '''def get_Others_response(user_query: str, dataset_address: str):
    """Use the shared CSVQA ReAct workflow for general and mixed CSV problems."""
    examples = build_formulation_examples("Others", user_query, k=3)
    prefix = f"""You are an expert optimization modeler.
Reference examples illustrate structure only; the original question and current data are authoritative.
{examples}
Return a concise symbolic model with sets, parameters, variable domains, the complete objective,
all constraints and exact source Data Mapping. Preserve all interacting decision families.
An explicitly complete entity list in the question is authoritative; otherwise use all required data.
For finite-horizon changes or ramps, an initial binary state does not supply a continuous initial level.
For recurring cyclic operation, preserve wrap-around coverage of shifts across the period boundary.
Minimum-down-time restrictions start on an on-to-off transition; allow an always-on sequence.
Preserve the question's complete summation scopes, units, additive constants and boundary conditions.
Mirror matrix entries only when symmetry is explicitly stated, checking identifiers and conflicting entries.
Distinguish axis captions and scalar-parameter rows from decision entities through their identifiers.
State query-requested filters and joins explicitly; a preview or example never defines the selected set.
"""
    return formulate_with_csvqa(
        user_query, dataset_address, "Others", CSVQA_PLANNED_PROMPTS["Others"],
        CSVQA_TOOL_DESCRIPTIONS["Others"], prefix, "",
    )
'''

FRAME_EXAMPLE = '''
Concrete runtime interface (replace placeholder table/column names using Data Mapping):
    frame = CSVQA_FRAMES["<table_id>"]
    for source_row, row in frame.iterrows():
        identifier = row["<identifier column>"]
        coefficient = float(row["<numeric column>"])
CSVQA_FRAMES entries are pandas DataFrames, not table dictionaries: they have no
"records" column or .records attribute. frame.to_dict("records") returns flat field
dictionaries; access row[exact_column], never row["values"]. source_row is frame.index,
not a source column unless the original CSV explicitly contains a column with that name.
Do not use row.values from itertuples as a source-record dictionary. Iterate with iterrows
or flat to_dict("records") dictionaries when source headers are not valid Python identifiers.
'''

DIRECT_SOURCE = '''
Read the exact source CSV paths in Source Schema at runtime with pandas. Preserve text
identifiers and complete source rows, then implement query-required joins and calculations.
No complete current CSV data is provided to this generation stage. Never invent coefficients
or encode assumed data values. Use only source columns and the mathematical model.
Try UTF-8-sig, UTF-8, GBK and Latin-1 only on decoding errors, matching the shared reader.
'''


def main():
    books = {method: json.loads((ROOT / name).read_text()) for method, name in NAMES.items()}
    full_original = books['full']
    legacy_node = next(n for n in ast.parse(''.join(full_original['cells'][27]['source'])).body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == 'LEGACY_CODE_INSTRUCTIONS')
    legacy = ast.literal_eval(legacy_node.value)
    changes = {}
    for method, book in books.items():
        changed = []
        for i, cell in enumerate(book['cells']):
            original = ''.join(cell['source'])
            s = original.replace('_v7', '_v8').replace('react-source-data-v7', 'react-source-data-v8')
            if cell['cell_type'] == 'code':
                s = s.replace('"Others": "legacy"', '"Others": "planned"')
                s = s.replace('Canonical CSV routes use source-derived records; Others retains its original flow.',
                              'Every CSV route uses the shared source-derived CSVQA workflow.')
                s = s.replace('for _route in ("TP", "AP", "FLP"):', 'for _route in ("TP", "AP", "FLP", "Others"):')
                # A tool requirement belongs in the shared agent, not conflicting route prefixes.
                s = s.replace('Call CSVQA exactly once and return an ABSTRACT model.', 'Return an ABSTRACT model.')
                s = s.replace('You MUST call CSVQA exactly once before the Final Answer and use all returned rows.',
                              'Use all current returned rows required by the original query.')
                s = s.replace('Call CSVQA at least once and return a complete numerical formulation. Retrieved\nInformation must contain every identifier and coefficient required by downstream code.',
                              'Return an ABSTRACT symbolic formulation and exact source Data Mapping. Keep\nnumerical identifiers and coefficients in the current Observation for downstream code.')
                if 'def _source_candidate(' in s:
                    s = replace_function(s, '_source_candidate', SOURCE_CANDIDATE)
                if 'def get_Others_response(' in s:
                    s = replace_function(s, 'get_Others_response', OTHERS)
                if 'def build_csvqa_components(' in s:
                    old = '            tables = _load_tables(dataset_address, file_indices=file_indices)'
                    s = s.replace(old,
                                  '            try:\n' + old.replace('            ', '                ', 1) + '\n'
                                  '            except ValueError as exc:\n'
                                  '                raise ReActProtocolError("csvqa_action_input_format") from exc')
                if 'PLANNED_CODE_INSTRUCTIONS =' in s:
                    for n in ast.parse(s).body:
                        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                            name = n.targets[0].id
                            if name == 'PLANNED_CODE_INSTRUCTIONS':
                                lines = s.splitlines(keepends=True)
                                s = ''.join(lines[:n.lineno-1]) + name + ' = ' + repr(ast.literal_eval(n.value) + FRAME_EXAMPLE) + '\n' + ''.join(lines[n.end_lineno:])
                                break
                    s = s.replace('Use numeric variable bounds or omit them; never pass None as lb or ub.',
                                  'Use numeric variable bounds or omit them; never pass None as lb or ub.\n'
                                  'For a variable unrestricted below, use lb=-gp.GRB.INFINITY; omitting lb means zero.\n'
                                  'Omit ub for its infinity default. Access decisions through their variable containers\n'
                                  'or m.getVars(); do not rely on getVarByName or formatted solver-name strings.')
                    if method == 'few_shot_only':
                        for n in ast.parse(s).body:
                            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'LEGACY_CODE_INSTRUCTIONS':
                                lines = s.splitlines(keepends=True)
                                s = ''.join(lines[:n.lineno-1]) + 'LEGACY_CODE_INSTRUCTIONS = ' + repr(legacy) + '\n' + ''.join(lines[n.end_lineno:])
                                break
                    s = s.replace('def _generate_code(output, route, original_query="", data_payload="", legacy_observation=""):',
                                  'def _generate_code(output, route, original_query="", data_payload="", legacy_observation="", *, direct_source=False):')
                    s = s.replace('parts.append(LEGACY_CODE_INSTRUCTIONS)',
                                  'parts.append(DIRECT_SOURCE_CODE_INSTRUCTIONS if direct_source else LEGACY_CODE_INSTRUCTIONS)')
                    s = s.replace('                          source_schema_only(_CURRENT_DATASET_ADDRESS))',
                                  '                          source_schema_only(_CURRENT_DATASET_ADDRESS), direct_source=True)')
                    s += '\nDIRECT_SOURCE_CODE_INSTRUCTIONS = ' + repr(DIRECT_SOURCE) + '\n'
                # Correct only separate-library LOTO semantic exclusion, not query-only modeling.
                s = s.replace('if "resource allocation" in text or "knapsack" in text:',
                              'if "resource allocation" in text:')
            if i == 6 and cell['cell_type'] == 'markdown':
                s = '## 3. CSV loading and route modes\n\nAll six CSV workflows use shared ReAct and planned source extraction. Few-shot Only deliberately substitutes a complete Python Observation without CSVQA. The original query-only route remains.\n'
            if i == 24 and cell['cell_type'] == 'markdown':
                s = '## 12. Other and mixed problems\n\nCSV cases use the same required-CSVQA ReAct agent, source interfaces and code generator as the canonical routes. Query-only cases retain the original ORLM_QA workflow.\n'
            if i == 26 and cell['cell_type'] == 'markdown':
                s = '## 13. Generate Python code\n\nCSV code uses exact current source DataFrames. Few-shot Only supplies source paths/schema without the complete current Observation. Query-only code remains self-contained.\n'
            if cell['cell_type'] == 'code':
                ast.parse(s)
            if s != original:
                changed.append(i)
                cell['source'] = s.splitlines(keepends=True)
                if cell['cell_type'] == 'code':
                    cell['execution_count'], cell['outputs'] = None, []
        changes[method] = changed
    # Save full first; descendants validate the exact final source identity.
    full_path = ROOT / NAMES['full']
    full_path.write_text(json.dumps(books['full'], ensure_ascii=False, indent=1) + '\n')
    full_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
    for method, book in books.items():
        if method == 'full':
            continue
        for cell in book['cells']:
            s = ''.join(cell['source'])
            s = re.sub(r'((?:BASE_NOTEBOOK_SHA256|LOTO_BASE_SHA256) = )"[a-f0-9]+"',
                       lambda m: m.group(1) + '"' + full_sha + '"', s)
            cell['source'] = s.splitlines(keepends=True)
        (ROOT / NAMES[method]).write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')
    (OUT / 'revision_scope_v8.json').write_text(json.dumps({
        'version': 'v8', 'full_sha256': full_sha, 'changed_cells': changes,
        'no_question_id_generation_patch': True, 'repair': False,
        'all_csv_routes_react_csvqa': True, 'query_only_formulation_preserved': True,
        'source_normalization': 'syntax/fence only; preserve generated variable names',
        'reference_objectives_and_input_data_modified': False,
        'ablation_launch_condition': 'Complete pass; no Objective Match decline for 101, Variants, each sheet or 101 class versus v7',
        '606_enabled': False,
    }, indent=2))
    print(json.dumps({'version': 'v8', 'changed_cells': changes, 'full_sha256': full_sha}, indent=2))


if __name__ == '__main__':
    main()
