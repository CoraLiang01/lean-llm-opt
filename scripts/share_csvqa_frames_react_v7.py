"""Expose source-preserving DataFrames at the common CSVQA/code boundary."""
import ast
import hashlib
import json
import re

from restore_react_revision_20261006 import ROOT, OUT, FILES, SOURCE_FILES, save, replace_function

FRAME_HELPER = '''def source_csvqa_frames(payload):
    """Create source-only DataFrames keyed by exact table_id; retain strings and row order."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    return {table["table_id"]: pd.DataFrame(
        [record["values"] for record in table["records"]], columns=table["columns"],
        index=[record["source_row"] for record in table["records"]],
    ) for table in data["tables"]}
'''

FRAME_INSTRUCTIONS = '''
Use the formulation for model structure and Data Mapping. At execution, CSVQA_FRAMES
is a dictionary from exact table_id to a pandas DataFrame of the original source fields.
Read every coefficient and entity field from CSVQA_FRAMES[table_id]. Values remain
original strings, including empty fields; explicitly convert required numeric columns.
Do not redefine these frames, copy coefficients into literals, or read external files.
Use exact column names and source IDs. Do not guess table/record dictionary layouts.
CSVQA_DATA remains available ONLY for metadata such as roles and matrix relationships.
Its top-level table objects have records; values belongs to a record, not to a table.
Do not access a table as a record or reparse structured source data as CSV text.
Preserve source row order, complete entity sets and both matrix axes. Select the exact
table_id from Data Mapping; roles may repeat. A copied example's entity list never
defines the current set without an explicit original-query restriction.
Optional row_id_mapping/column_id_mapping in matrix metadata maps raw labels to entity
IDs. Apply it only when supplied; otherwise use exact raw IDs. Preserve all coefficients.
Without a continuous pre-horizon value, start change/ramp constraints at the second
period; an initial binary state does not supply a continuous period-zero value.
Only parse individual source fields when those fields themselves contain structured text.
'''


def main():
    books = {}
    for method, name in FILES.items():
        book = json.loads((OUT / 'notebook_snapshots/v6' / SOURCE_FILES[method]).read_text())
        for cell in book['cells']:
            source = ''.join(cell['source'])
            source = source.replace('_v6', '_v7').replace('react-source-data-v6', 'react-source-data-v7')
            if cell['cell_type'] == 'code':
                nodes = ast.parse(source).body
                for node in nodes:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PLANNED_CODE_INSTRUCTIONS' for t in node.targets):
                        lines = source.splitlines(keepends=True)
                        source = (''.join(lines[:node.lineno-1]) + 'PLANNED_CODE_INSTRUCTIONS = ' + repr(FRAME_INSTRUCTIONS) + '\n' +
                                  ''.join(lines[node.end_lineno:]))
                        break
            if 'def execute_pipeline_case(' in source and 'CSVQA_DATA = {pprint.pformat' in source:
                old = 'source = f"CSVQA_DATA = {pprint.pformat(data, sort_dicts=False, width=120)}\\n{source}"'
                new = ('source = (f"CSVQA_DATA = {pprint.pformat(data, sort_dicts=False, width=120)}\\n"\n'
                       '                      "import pandas as pd\\n"\n'
                       '                      "CSVQA_FRAMES = {t[\\\"table_id\\\"]: pd.DataFrame("\n'
                       '                      "[r[\\\"values\\\"] for r in t[\\\"records\\\"]], columns=t[\\\"columns\\\"], "\n'
                       '                      "index=[r[\\\"source_row\\\"] for r in t[\\\"records\\\"]]) "\n'
                       '                      "for t in CSVQA_DATA[\\\"tables\\\"]}\\n" + source)')
                assert old in source
                source = source.replace(old, new)
            if 'def invoke_react_protocol(' in source:
                source = source.replace('            agent = agent_factory()',
                    '            agent = agent_factory()\n'
                    '            original_template = agent.agent.llm_chain.prompt.template')
                source = source.replace('            result = agent.invoke(query)',
                    '            try:\n'
                    '                result = agent.invoke(query)\n'
                    '            finally:\n'
                    '                agent.agent.llm_chain.prompt.template = original_template')
            if 'def build_csvqa_components(' in source:
                source = source.replace('            request = json.loads(tool_query)',
                    '            try:\n'
                    '                request = json.loads(tool_query)\n'
                    '            except json.JSONDecodeError as exc:\n'
                    '                raise ReActProtocolError("csvqa_action_input_format") from exc')
            if cell['cell_type'] == 'markdown' and '## 14. Execute and assemble results' in source:
                source += ('\nStructured CSVQA records also have a source-only CSVQA_FRAMES interface keyed by table_id. '
                           'This is deterministic data transfer, not a generated-code repair.\n')
            cell['source'] = source.splitlines(keepends=True)
        book['cells'][7]['source'] += ['\n'] + FRAME_HELPER.splitlines(keepends=True)
        books[method] = book
    save(ROOT / FILES['full'], books['full'])
    full_sha = hashlib.sha256((ROOT / FILES['full']).read_bytes()).hexdigest()
    for method, book in books.items():
        if method == 'full':
            continue
        for cell in book['cells']:
            source = ''.join(cell['source'])
            field = 'BASE_NOTEBOOK_SHA256' if method in {'rag_only', 'few_shot_only'} else 'LOTO_BASE_SHA256'
            source = re.sub(field + r' = "[a-f0-9]+"', f'{field} = "{full_sha}"', source)
            cell['source'] = source.splitlines(keepends=True)
        save(ROOT / FILES[method], book)
    (OUT / 'revision_scope_v7.json').write_text(json.dumps({
        'version':'v7', 'files':FILES, 'architecture':'Original ReAct with user-authorized protocol restarts',
        'data_interface':'CSVQA_FRAMES[table_id], exact original strings/columns/source row order',
        'data_repair':False, 'generated_code_repair':False, 'objective_based_retry':False,
        'csvqa_minimum_calls':1, 'multiple_csvqa_calls':True,
        'format_restart_includes_csvqa_json_input':True,
        'cached_prompt_template_restored_after_invocation':True, '606_enabled':False,
    },indent=2))
    print('Built v7: source-only DataFrame handoff, no generated-code edits or objective-based retries.')


if __name__ == '__main__':
    main()
