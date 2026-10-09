"""Validate legacy extraction against source rows, retaining ReAct and legacy mode."""
import hashlib
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/review_0927_20261006'
SOURCE=OUT/'candidates/full_review_v4.ipynb'
DEST=OUT/'candidates/full_review_v6.ipynb'

VALIDATOR='''def validate_legacy_source_rows(observation, documents):
    """Accept only exact source-row evidence, including original fields and multiplicity."""
    from collections import Counter
    original = [json.loads(document.page_content) for document in documents]
    observed = json.loads(str(observation))
    if not isinstance(observed, list) or (original and not observed):
        raise ValueError("Legacy extraction must return a nonempty source-row array")
    def signature(row):
        if not isinstance(row, dict) or not isinstance(row.get("values"), dict):
            raise ValueError("Legacy extraction row is missing its original field mapping")
        return (str(row.get("source", "")), tuple(sorted(
            (str(column), str(value)) for column, value in row["values"].items())))
    available = Counter(signature(row) for row in original)
    for row in observed:
        key = signature(row)
        if available[key] <= 0:
            raise ValueError("Legacy extraction contains altered/derived fields, a foreign row, or an extra duplicate")
        available[key] -= 1
    return str(observation)


'''


def main():
    book=json.loads(SOURCE.read_text());edits=[]
    def edit(i,old,new):
        s=''.join(book['cells'][i]['source']);assert s.count(old)==1,(i,old[:80])
        book['cells'][i]['source']=s.replace(old,new).splitlines(True)
        edits.append({'cell':i,'old':old,'new':new})
    edit(11,'def build_csvqa_components(',VALIDATOR+'def build_csvqa_components(')
    edit(11,'        return str(result)\n',
         '        try:\n'
         '            observation = validate_legacy_source_rows(result, documents)\n'
         '            return observation, "LEGACY_VALIDATED_SOURCE_ROWS", None, str(result)\n'
         '        except (KeyError, TypeError, ValueError) as exc:\n'
         '            # Recover original evidence; selection remains the modeling agent\'s query-based task.\n'
         '            observation = json.dumps([json.loads(d.page_content) for d in documents], ensure_ascii=False)\n'
         '            print(f"[Rejected legacy extraction]: {exc}\\n{result}")\n'
         '            return observation, "LEGACY_SOURCE_VALIDATION_FALLBACK", str(exc), str(result)\n')
    edit(11,'            observation = rag_answer(tool_query)\n',
         '            observation, status, fallback_reason, legacy_output = rag_answer(tool_query)\n'
         '            if fallback_reason:\n'
         '                errors.append(fallback_reason)\n')
    edit(11,'            "validation_errors": errors, "fallback_reason": fallback_reason,\n',
         '            "validation_errors": errors, "fallback_reason": fallback_reason,\n'
         '            "legacy_extraction_output": legacy_output if not planned else None,\n')
    edit(15,'    prefix += QUERY_SEMANTICS_GUIDANCE\n',
         '    prefix += QUERY_SEMANTICS_GUIDANCE\n'
         '    prefix += "\\nUse the ReAct protocol: after reasoning/tools, begin the final model with the literal marker Final Answer:. Any complete source fallback is evidence, not automatic selection: apply only the original query\'s subset and eligibility rules in the model."\n')
    edit(25,'        code_gen_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n',
         '        code_gen_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n'
         '        code_gen_template += "\\nDataFrame.columns is an Index, not a dictionary: test column_name in df.columns, then access df[column_name]. Do not call columns.get(). Gurobi comparisons are TempConstr objects: pass them to addConstr, never multiply them or pass them to quicksum. For a positive-quantity prerequisite, use the already linked binary selection flags: selected_item <= selected_prerequisite, without a quantity ratio."\n')
    edit(29,'        model_text = formulation["formulation"]\n',
         '        record["csvqa_trace"] = json.dumps(formulation.get("trace", {}), ensure_ascii=False)\n'
         '        model_text = formulation["formulation"]\n')
    src=''.join(book['cells'][33]['source'])
    assert 'RECORD_ARTIFACT_FILES = {' in src
    edit(33,'RECORD_ARTIFACT_FILES = {',
         'RECORD_ARTIFACT_FILES = {\n    "csvqa_trace": "csvqa_trace.json",')
    edit(3,'outputs/optimization_0927_20261006/full_review_v4','outputs/optimization_0927_20261006/full_review_v6')
    book['cells'][3]['source']=''.join(book['cells'][3]['source']).replace(str(SOURCE),str(DEST)).splitlines(True)
    book['metadata']['review_candidate']['revision']='review-v6'
    book['metadata']['review_candidate']['source_validation']='Exact source-row field/value/multiplicity check; complete original evidence fallback on invalid extraction, with reason retained; no change to mode, ReAct or query selection. No gold data.'
    for i,c in enumerate(book['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
    content=json.dumps(book,ensure_ascii=False,indent=1)+'\n'
    if DEST.exists() and DEST.read_text()!=content:
        assert not (ROOT/'outputs/optimization_0927_20261006/full_review_v6/frozen_notebook.ipynb').exists(),'Use a new revision'
        draft=OUT/'candidate_drafts'/('full_review_v6_draft_'+hashlib.sha256(DEST.read_bytes()).hexdigest()[:12]+'.ipynb')
        draft.write_bytes(DEST.read_bytes())
    DEST.write_text(content)
    nbformat.validate(nbformat.read(DEST,as_version=4))
    (OUT/'patch_v6.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2))
    print(json.dumps({'candidate':str(DEST),'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'edits':len(edits)},indent=2))


if __name__=='__main__':main()
