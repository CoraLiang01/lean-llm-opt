"""Repair protocol, raw-table eligibility evidence and invalid generated API guidance."""
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/review_0927_20261006'
SOURCE = OUT / 'candidates/full_review_v4.ipynb'
DEST = OUT / 'candidates/full_review_v5.ipynb'


def main():
    book=json.loads(SOURCE.read_text()); edits=[]
    def edit(i, old, new):
        s=''.join(book['cells'][i]['source']);assert s.count(old)==1,(i,old[:60])
        book['cells'][i]['source']=s.replace(old,new).splitlines(True)
        edits.append({'cell':i,'old':old,'new':new})
    edit(15, '    prefix += QUERY_SEMANTICS_GUIDANCE\n',
         '    prefix += QUERY_SEMANTICS_GUIDANCE\n'
         '    prefix += "\\nUse the ReAct protocol. After tool use and reasoning, start the final model with the literal marker Final Answer:. Return each model section and each required parameter table once, without duplicating the complete model or repeating the same data in both JSON and Markdown."\n')
    old="csvqa_system_prompt = 'Retrieve the complete assignment data needed by the query, preserving IDs and costs together with eligibility, availability and qualification fields. For ordered qualifications, use the order stated in the query, never alphabetical order. Preserve the fields needed to verify each assignment pair; do not label an unverified pair eligible. Context: {context}'"
    new="csvqa_system_prompt = 'Retrieve all query-needed source assignment tables separately: entity availability/qualifications, task requirements and original cost matrix or offer rows. Keep original source paths, column names and cell values. Do not join tables into new rows, derive qualification fields or precompute an eligible-offer list. The modeling stage applies eligibility using the complete original evidence. Context: {context}'"
    edit(21,old,new)
    edit(25, '        abstract_model_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n',
         '        abstract_model_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n'
         '        abstract_model_template += "\\nEmit one compact plan; each variable, parameter mapping and constraint family appears once. Shared preprocessing rules are stated once for all applicable tables. Stop at the end marker; no repeated restatement or worked data calculations."\n')
    edit(25, '        code_gen_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n',
         '        code_gen_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n'
         '        code_gen_template += "\\nDataFrame.columns is an Index, not a dictionary: test column_name in df.columns, then access df[column_name]. Test a table tag in the column values, never by calling columns.get(). Reuse a common preprocessing helper for tables with the same event-selection rule; avoid near-identical filter functions for each table. Gurobi comparisons produce TempConstr objects, not numeric expressions: pass them to addConstr, never multiply them or pass them to quicksum. A positive-selection prerequisite uses selected_item <= selected_prerequisite with their established quantity links, not a quantity ratio."\n')
    edit(27, 'CSV_SOLVER_INSTRUCTIONS += QUERY_SEMANTICS_GUIDANCE\n',
         'CSV_SOLVER_INSTRUCTIONS += QUERY_SEMANTICS_GUIDANCE\n'
         'CSV_SOLVER_INSTRUCTIONS += "\\nGurobi comparisons are constraints, not numeric expressions. Pass them to addConstr; do not multiply them or sum a comparison. A positive-selection prerequisite uses selected_item <= selected_prerequisite with valid quantity links. If reading pandas tables, test column_name in df.columns before accessing df[column_name]; Index has no dictionary get method."\n')
    edit(3, 'outputs/optimization_0927_20261006/full_review_v4', 'outputs/optimization_0927_20261006/full_review_v5')
    book['cells'][3]['source']=''.join(book['cells'][3]['source']).replace(str(SOURCE),str(DEST)).splitlines(True)
    book['metadata']['review_candidate']['revision']='review-v5'
    book['metadata']['review_candidate']['protocol_fixes']='ReAct end marker; original AP table rows without derived joins; compact plans; valid Index and Gurobi comparison APIs. No gold model or case-specific branches.'
    for i,c in enumerate(book['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
    content=json.dumps(book,ensure_ascii=False,indent=1)+'\n'
    if DEST.exists():assert DEST.read_text()==content,'Use a new revision'
    else:DEST.write_text(content)
    nbformat.validate(nbformat.read(DEST,as_version=4))
    (OUT/'patch_v5.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2))
    print(json.dumps({'candidate':str(DEST),'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'edits':len(edits)},indent=2))


if __name__=='__main__':main()
