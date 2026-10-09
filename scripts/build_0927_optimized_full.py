"""Apply small common prompt/format/transport fixes to the frozen full baseline."""
import copy
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/optimization_0927_20261006'
BEFORE = OUT/'source_snapshots/before/full_before.ipynb'

SEMANTICS = (
    'Preserve the current query\'s conditional versus unconditional bounds. '
    'A fixed activation fee does not make a quantity lower bound conditional: '
    'apply an unconditional bound even when a category has no selected item. '
    'Use activation-conditioned bounds only if the query explicitly makes them conditional.'
)
CODE_SAFETY = (
    '\nGive each decision-variable container a distinct name ending in _vars; never reuse it '
    'as a loop variable, scalar key, parameter or row object. In a generator expression, '
    'index the decision container with the locally bound key, not with a shadowed container. '
    'quicksum takes an iterable; a scalar constant is used directly.\n'
    'Read source CSVs with dtype=str and keep_default_na=False, then explicitly convert '
    'the required numeric fields. Preserve blank forbidden entries and exact identifiers.\n'
    + SEMANTICS + '\n'
)
INTERFACE = (
    '\nMandatory result interface: expose exactly one solved, live Gurobi model at module '
    'scope as m. A function that creates a model must explicitly return m, and its caller '
    'must assign that result to m. Never assign optimize()\'s return to m. Do not dispose '
    'the model or leave its context manager before the harness extracts the result. '
    'Printed objectives cannot replace this model object.\n'
)


def source(book, i):
    return ''.join(book['cells'][i]['source'])


def put(book, i, text):
    book['cells'][i]['source'] = text.splitlines(keepends=True)


def main():
    book = json.loads(BEFORE.read_text())
    put(book, 2, source(book, 2).replace('import openai\n', 'import openai\nimport httpx\n'))
    s = source(book, 5)
    s = '''HTTP_RETRY_EVENTS = []


def _record_http_request(request):
    index = int(request.headers.get("x-stainless-retry-count", "0"))
    if index:
        HTTP_RETRY_EVENTS.append({"endpoint": request.url.path, "retry_index": index})


@lru_cache(maxsize=1)
def _get_http_client():
    return httpx.Client(timeout=180, event_hooks={"request": [_record_http_request]})


''' + s
    s = s.replace('max_retries=0, timeout=timeout,',
                  'max_retries=1, http_client=_get_http_client(), timeout=timeout,')
    s = s.replace('openai_api_key=require_api_key(), max_retries=0)',
                  'openai_api_key=require_api_key(), max_retries=1, http_client=_get_http_client())')
    put(book, 5, s)
    s = source(book, 11)
    s = s.replace('page_content=json.dumps({"values": values}, ensure_ascii=False)',
                  'page_content=json.dumps({"source": str(table["path"]), "values": values}, ensure_ascii=False)')
    s = s.replace('("system", system_prompt),', '''("system", system_prompt +
                "\\nResponse format: return one JSON array, with one object per source row. "
                "Each object has source (the exact supplied source path) and values "
                "(a mapping of exact column names to original source values). "
                "Preserve source order, identifiers, signs and blank values. "
                "Do not add prose, Markdown, numbered headings or ellipses. "
                "An unrelated column does not create a model coefficient or constraint."),''')
    put(book, 11, s)
    s = source(book, 15)
    s = s.replace('    agent = initialize_agent(', '    prefix += ' + repr('\n'+SEMANTICS) + '\n    agent = initialize_agent(', 1)
    put(book, 15, s)
    s = source(book, 25)
    s = s.replace('        abstract_prompt = PromptTemplate(',
        '        abstract_model_template += ' + repr('\n'+SEMANTICS+
            '\nReturn exactly one concise abstract plan with the seven requested sections. '
            'Describe each constraint family once. Use symbolic index sets and exact field mappings; '
            'do not enumerate source rows, repeat examples, or emit code. Preserve every required '
            'variable, domain, objective term and boundary condition.') + '\n        abstract_prompt = PromptTemplate(', 1)
    s = s.replace('        code_gen_prompt = PromptTemplate(',
        '        code_gen_template += ' + repr(CODE_SAFETY+INTERFACE) + '\n        code_gen_prompt = PromptTemplate(', 1)
    put(book, 25, s)
    s = source(book, 27)
    s = s.replace('return {"source": source, "values":', 'return {"source": source or str(row.get("source", "")), "values":', 1)
    s = s.replace('    parts.extend([CSV_SOLVER_INSTRUCTIONS])',
                  '    parts.extend([CSV_SOLVER_INSTRUCTIONS])')
    marker = '    response = make_llm().invoke([HumanMessage(content="\\n\\n".join(parts))])'
    assert marker in s
    s = s.replace(marker, '    parts.append(' + repr(CODE_SAFETY+INTERFACE) + ')\n'+marker)
    s += '\nCSV_ROUTE_HINT["AP"] += ' + repr(
        ' Eligibility is a decision-pair mask. A nonblank cost for an ineligible pair '
        'is valid source data: exclude or fix that decision to zero; do not reject '
        'the dataset because such a pair has a listed cost.')+'\n'
    put(book, 27, s)
    s = source(book, 29)
    s = s.replace('    query, address = record["query"], record["dataset_address"]',
                  '    HTTP_RETRY_EVENTS.clear()\n    query, address = record["query"], record["dataset_address"]')
    s = s.replace('        record.update(final_objective=',
                  '        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))\n        record.update(final_objective=')
    s = s.replace('    except Exception as exc:\n        exc.pipeline_context = record',
                  '    except Exception as exc:\n        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))\n        exc.pipeline_context = record')
    put(book, 29, s)
    folder = OUT/'source_snapshots/v1'; folder.mkdir(parents=True, exist_ok=True)
    name = 'LEAN_LLM_OPT_4.1_Large-scale_0927_optimized.ipynb'
    s = source(book, 3).replace('full_before.ipynb', name)
    s = s.replace('outputs/ablation_loto_0927/examples_and_route_v1', 'outputs/optimization_0927_20261006/full_v1')
    put(book, 3, s)
    book['cells'][0]['source'] = ['# 0927 full model — minimal revision v1\n',
        'Original ReAct agents and route data modes retained. Only common prompt/format and transport fixes.\n']
    for i, c in enumerate(book['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), name+f':cell{i}', 'exec')
            c['outputs'] = []; c['execution_count'] = None
    book['metadata']['experiment_provenance'].update(revision='v1', baseline_sha256=hashlib.sha256(BEFORE.read_bytes()).hexdigest())
    path = folder/name; path.write_text(json.dumps(book, ensure_ascii=False, indent=1)+'\n')
    nbformat.validate(nbformat.read(path, as_version=4))
    scope = {'revision': 'v1', 'base': str(BEFORE), 'candidate': str(path),
             'changed_code_cells': [2,5,11,15,25,27,29],
             'changes': ['shared HTTP pool and one SDK transport retry, with recorded retry events',
                 'source-labelled JSON legacy evidence and preserve its source during parsing',
                 'conditional/unconditional constraint guidance', 'concise Others abstract plan',
                 'non-shadowing variable names and raw CSV strings', 'AP eligibility mask guidance',
                 'explicit model-object interface guidance'],
             'unchanged': ['ReAct framework', 'route data modes: only NRM planned',
                 'model snapshot and sampling parameters', 'solver instructions and execution function',
                 'objective tolerance', 'reference examples and benchmark data'],
             'evidence_cases': ['Variant1','Variant13','Variant17','Variant27','Variant28','Variant30','Variant32','Variant35','50pct-S1/OR-007','50pct-S1/OR-014']}
    (OUT/'revision_scope_v1.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2))
    print(path)


if __name__ == '__main__':
    main()
