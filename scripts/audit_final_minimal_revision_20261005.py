"""Read-only provenance, experimental-boundary and parameter audit; no API calls."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/minimal_revision_20261005'
VERSIONS = {'full':'v4','rag_only':'v4','few_shot_only':'v5','examples_only':'v4','examples_and_route':'v4'}
NAMES = {'full':'LEAN_LLM_OPT_4.1_Large-scale.ipynb',
         'rag_only':'Ablation_Study_Large_Scale_Or_RAG_Only.ipynb',
         'few_shot_only':'Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb',
         'examples_only':'LOTO_Examples_Only_GPT4.1_Large-scale.ipynb',
         'examples_and_route':'LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(nb):
    return '\n\n'.join(''.join(c['source']) for c in nb['cells'])


def query_only_function(nb):
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        for node in ast.parse(text).body:
            if isinstance(node,ast.FunctionDef) and node.name == 'get_others_without_CSV_response':
                return ast.get_source_segment(text,node)
    raise ValueError('Missing query-only function')


def main(complete=False):
    sources, inputs, counts, records, folds = {}, {}, {}, [], []
    parameters, parameter_cases, unparseable = defaultdict(set), [], []
    for method, version in VERSIONS.items():
        folder = OUT / f'{method}_{version}'
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['model']=='gpt-4.1-2025-04-14'
        assert manifest['rel_tol']==manifest['abs_tol']==1e-4
        assert manifest['gurobi_threads']==2 and manifest['api_max_retries']==2
        assert not any(manifest[x] for x in ['model_repair','code_repair','truncation_retry','parsing_correction'])
        for path, digest in manifest['inputs'].items():
            assert path not in inputs or inputs[path]==digest, (method,path)
            inputs[path]=digest
        expected=sum(manifest['sizes'].values())
        pairs=[(p,json.loads(p.read_text())) for p in folder.glob('**/attempts/*/result.json')]
        counts[method]={'version':version,'expected':expected,'recorded':len(pairs)}
        keys=[str(p.parent.relative_to(folder)) for p,_ in pairs]
        assert len(set(keys))==len(keys)
        if complete:
            assert len(pairs)==expected,(method,len(pairs),expected)
        book=json.loads((ROOT/NAMES[method]).read_text())
        frozen=json.loads((folder/'frozen_notebook.ipynb').read_text())
        difference=[i for i,(x,y) in enumerate(zip(book['cells'],frozen['cells'])) if x['source']!=y['source']]
        expected_difference = [38,39] if method=='full' else [39] if method.startswith('examples') else []
        assert difference==expected_difference,(method,difference)
        if method.startswith('examples'):
            old_source=''.join(frozen['cells'][39]['source'])
            assert old_source.replace('if "resource allocation" in text:', 'if "resource allocation" in text or "knapsack" in text:')==''.join(book['cells'][39]['source'])
            assert all(r['dataset_address'].strip() for _,r in pairs)
        assert not re.search('[\u4e00-\u9fff]',source(book))
        for cell in book['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        sources[NAMES[method]]={'syntax':'PASS','English_notebook_source':'PASS','sha256':sha(ROOT/NAMES[method]),
                               'frozen_sha256':sha(folder/'frozen_notebook.ipynb'),'changed_cell_indices':difference}
        for path,r in pairs:
            assert r.get('repair_count',0)==r.get('retry_count',0)==0
            events=r.get('api_retry_events',[])
            if isinstance(events,str):events=json.loads(events)
            assert isinstance(events,list) and r.get('api_retry_count',0)==len(events),(method,path)
            assert r['cache_source']=='computed', (method,path,r.get('cache_source'))
            assert r['notebook_source_sha256']==manifest['notebook_sha256']
            assert r['base_notebook_sha256']==manifest['base_sha256']
            marker=json.loads(path.with_name('attempt.json').read_text())
            assert marker['state']=='finished'
            records.append((method,r))
            if method.startswith('examples'):
                fold=json.loads(path.with_name('fold_manifest.json').read_text())
                assert fold['held_out_type']==r['true_label']
                assert fold['remaining_counts'].get(r['true_label'],0)==0
                assert all(('FLP' if x['type']=='UFLP' else x['type'])==r['true_label'] for x in fold['removed_examples'])
                disabled=fold['disabled_workflow_route']
                executed=r.get('pipeline_stage') not in ('classification','route_selection','setup')
                assert not (disabled and r.get('assigned_route')==disabled and executed)
                folds.append({'method':method,'id':r['problem_id'],'held_out_type':r['true_label'],
                              'target_examples_remaining':0,'disabled_route':disabled,
                              'assigned_route':r.get('assigned_route'),'disabled_route_executed':False})
            code=r.get('solve_code','')
            if not code:continue
            try:tree=ast.parse(code)
            except SyntaxError:
                unparseable.append({'method':method,'case':r['problem_id']});continue
            settings=[]
            for node in ast.walk(tree):
                name,value=None,None
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Attribute) and target.value.attr.casefold()=='params':
                            name=target.attr;value=ast.unparse(node.value)
                elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='setParam' and len(node.args)>=2:
                    name=ast.literal_eval(node.args[0]) if isinstance(node.args[0],ast.Constant) else '<dynamic>';value=ast.unparse(node.args[1])
                if name:
                    settings.append([name,value]);parameters[name].add(value)
            parameter_cases.append({'method':method,'version':version,'case':r['problem_id'],'artifact':str(path.parent.relative_to(OUT)),'settings':settings})
    changed_inputs=[path for path,digest in inputs.items() if sha(path)!=digest]
    assert not changed_inputs,changed_inputs
    baseline=json.loads((OUT/'full_v4/frozen_notebook.ipynb').read_text())
    backup=json.loads((OUT/'backups'/NAMES['full']).read_text())
    original=query_only_function(backup).replace('handle_parsing_errors=True','handle_parsing_errors=False')
    assert query_only_function(baseline)==original
    sources['full_query_only']={'unchanged_except_disabling_parser_correction':'PASS','benchmark_API_performance':'not evaluated; all 101 cases have CSV inputs'}
    full_by_id={r['problem_id']:r for m,r in records if m=='full' and r['problem_id'].startswith('OR-') and not r.get('experiment_mode','').startswith('columns')}
    # Only the automatic full case folders are classification-cache sources.
    full_by_id={json.loads(p.read_text())['problem_id']:json.loads(p.read_text()) for p in (OUT/'full_v4/automatic/attempts').glob('*/result.json')}
    for m,r in records:
        if m in ('rag_only','few_shot_only'):
            assert r['predicted_label']==full_by_id[r['problem_id']]['predicted_label']
            assert r['assigned_route']==full_by_id[r['problem_id']]['assigned_route']
            assert r['classification_source']=='final_full_cache'
    assert set(parameters)<= {'MIPGap'},dict(parameters)
    assert all(value=='0.0001' for value in parameters.get('MIPGap',[])),dict(parameters)
    initial=json.loads((OUT/'initial_manifest.json').read_text())
    assert all(sha(ROOT/p)==v['sha256'] for p,v in initial.items())
    totals={m:dict(repairs=sum(r.get('repair_count',0) for mm,r in records if mm==m),
                   pipeline_retries=sum(r.get('retry_count',0) for mm,r in records if mm==m),
                   transport_retries=sum(r.get('api_retry_count',0) for mm,r in records if mm==m),
                   recorded_fallbacks=sum(r.get('fallback_count',0) for mm,r in records if mm==m)) for m in VERSIONS}
    state='complete' if all(v['recorded']==v['expected'] for v in counts.values()) else 'incomplete'
    (OUT/'final_input_and_attempt_audit.json').write_text(json.dumps({'state':state,'api_calls':0,'checked_input_files':len(inputs),
        'modified_inputs':changed_inputs,'finished_final_cases':len(records),'counts':counts,
        'all_final_results_fresh_computation':True,'ablation_classifications_identical_to_full':True,
        'model_and_pipeline_repairs_or_retries':0,'duplicate_case_attempts':0,'usage_totals':totals},indent=2))
    (OUT/'final_source_verification.json').write_text(json.dumps(sources,indent=2))
    (OUT/'final_loto_fold_audit.json').write_text(json.dumps({'state':state,'disabled_route_executions':0,'folds':folds},indent=2))
    (OUT/'solver_parameter_audit.json').write_text(json.dumps({'state':state,'explicit_settings':{k:sorted(v) for k,v in parameters.items()},
        'global_threads':2,'unparseable_programs':unparseable,'cases':parameter_cases},indent=2))
    print(json.dumps({'state':state,'counts':counts,'inputs_unchanged':len(inputs),'usage':totals},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    main(parser.parse_args().require_complete)
