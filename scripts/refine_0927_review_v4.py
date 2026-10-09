"""Add general query-semantics contracts from observed modeling errors, without gold data."""
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
SOURCE = OUT / "candidates/full_review_v3.ipynb"
DEST = OUT / "candidates/full_review_v4.ipynb"

SEMANTICS = '''
The original query is authoritative when a reference example or intermediate plan conflicts with it.
Eligibility or authorization permits a choice; it does not force selection. When an integer
quantity may be zero or within [L,U], use a binary selected flag with L*selected <= quantity
<= U*selected, and enforce positive quantity iff selected (quantity >= selected when L is zero).
An activation flag means iff its stated condition holds: an item flag iff positive quantity;
a group flag iff at least one member is selected (member_flag <= group_flag for every member,
group_flag <= sum(member_flags)); a pair flag iff both members are selected (the two upper
bounds and the lower bound). Enforce both directions regardless of fee/bonus signs.
Keep unconditional aggregate bounds unconditional. Do not add restrictions, lazy constraints,
fixed-zero decisions, or mandatory selections without support in the current query.
'''

TABLES = '''
Identify logical tables from actual schema fields and table tags, never a guessed file index.
A logical table may span several files: collect all its rows before query-directed selection,
joining or aggregation. Apply a query-specified revision/as-of/deletion/retransmission rule to
EVERY applicable table, before numeric conversion or sums. Compare revisions numerically,
retain the latest eligible event by the full stated record key, then remove its tombstone.
Do not convert blank fields in deleted or other-table rows into numeric parameters, and do
not silently coerce required malformed numbers to zero. Do not collapse distinct signed
components into a single last-write dictionary entry when the query requires their sum.
Keep already consistent opaque business IDs as keys. Use a lookup only when crossing key
namespaces; a display label is not a replacement for a working canonical key.
Convert all quantities in a resource comparison to the same stated unit, on both sides.
'''


def main():
    book = json.loads(SOURCE.read_text())
    edits = []
    def edit(index, old, new):
        source = "".join(book["cells"][index]["source"])
        assert source.count(old) == 1, (index, old[:80])
        book["cells"][index]["source"] = source.replace(old, new).splitlines(True)
        edits.append({"cell": index, "old": old, "new": new})

    edit(15, 'def formulate_with_csvqa(',
         'QUERY_SEMANTICS_GUIDANCE = ' + repr(SEMANTICS) + '\n\n\ndef formulate_with_csvqa(')
    edit(15, '    agent = initialize_agent(\n        tools=[qa_tool]',
         '    prefix += QUERY_SEMANTICS_GUIDANCE\n    agent = initialize_agent(\n        tools=[qa_tool]')
    edit(21, "csvqa_system_prompt = 'Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead.Context: {context}'",
         "csvqa_system_prompt = 'Retrieve the complete assignment data needed by the query, preserving IDs and costs together with eligibility, availability and qualification fields. For ordered qualifications, use the order stated in the query, never alphabetical order. Preserve the fields needed to verify each assignment pair; do not label an unverified pair eligible. Context: {context}'")
    edit(21, '            When you need to retrieve information from the CSV file, use the provided tool.',
         '            Verify every assignment pair against ALL query eligibility conditions. For ordered\n'
         '            qualifications, construct ranks from the query-stated order, not lexical comparison.\n'
         '            Keep qualification and availability fields with the cost parameters so the code can\n'
         '            enforce eligibility; a listed cost alone does not make a pair eligible.\n\n'
         '            When you need to retrieve information from the CSV file, use the provided tool.')
    edit(25, 'def get_Others_response(',
         'SOURCE_TABLE_GUIDANCE = ' + repr(TABLES) + '\n\n\ndef get_Others_response(')
    edit(25, '        abstract_prompt = PromptTemplate(',
         '        abstract_model_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n        abstract_prompt = PromptTemplate(')
    edit(25, '        code_gen_prompt = PromptTemplate(',
         '        code_gen_template += QUERY_SEMANTICS_GUIDANCE + SOURCE_TABLE_GUIDANCE\n        code_gen_prompt = PromptTemplate(')
    edit(27, 'Preserve the formulation\'s variable domains. Return code only, without explanations',
         'Preserve variable domains consistent with the original query; the original query takes\nprecedence over conflicting intermediate wording. Return code only, without explanations')
    edit(27, '\n\ndef retrieve_csv_code_example(',
         '\nCSV_SOLVER_INSTRUCTIONS += QUERY_SEMANTICS_GUIDANCE\n\n\ndef retrieve_csv_code_example(')
    edit(27, "CSV_ROUTE_HINT[\"AP\"] += ' Eligibility is a decision-pair mask. A nonblank cost for an ineligible pair is valid source data: exclude or fix that decision to zero; do not reject the dataset because such a pair has a listed cost.'",
         "CSV_ROUTE_HINT[\"AP\"] += ' Eligibility is a decision-pair mask. A nonblank cost for an ineligible pair is valid source data: exclude or fix that decision to zero; do not reject the dataset because such a pair has a listed cost. Evaluate query-stated availability and qualification rules from the supplied eligibility fields, using the stated ordinal rank instead of alphabetical order.'")
    edit(3, 'outputs/optimization_0927_20261006/full_review_v3', 'outputs/optimization_0927_20261006/full_review_v4')
    cfg = ''.join(book['cells'][3]['source']).replace(str(SOURCE), str(DEST))
    book['cells'][3]['source'] = cfg.splitlines(True)
    book['metadata']['review_candidate']['revision'] = 'review-v4'
    book['metadata']['review_candidate']['semantic_fixes'] = 'Optional selection; bidirectional activation; schema-based table collection; selection before conversion; eligibility ranks from query. No case IDs or gold models.'
    for i,c in enumerate(book['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'cell{i}', 'exec')
    content = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    if DEST.exists():
        assert DEST.read_text() == content, 'Use a new revision instead of overwriting a candidate'
    else:
        DEST.write_text(content)
    nbformat.validate(nbformat.read(DEST, as_version=4))
    (OUT/'patch_v4.json').write_text(json.dumps(edits, ensure_ascii=False, indent=2))
    print(json.dumps({'candidate':str(DEST), 'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(), 'edits':len(edits)}, indent=2))

if __name__ == '__main__':
    main()
