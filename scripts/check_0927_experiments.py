"""Check component boundaries without making model calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
NAMES={'rag_only':'Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb',
       'few_shot_only':'Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb',
       'examples_and_route':'LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb'}

def namespace(path):
    ns={'__name__':'__evaluation__'}
    book=json.loads(Path(path).read_text())
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        for i,c in enumerate(book['cells']):
            if c['cell_type']=='code':
                s=''.join(c['source'])
                # Base notebook experiment switches are on; definitions only for the base.
                if Path(path).name.startswith('LEAN_') and i>35: continue
                exec(compile(s,f'{path}:cell{i}','exec'),ns)
    ns['NOTEBOOK_PATH']=Path(path).resolve()
    return ns


def main():
    import os
    os.chdir(ROOT)
    base=namespace(ROOT/'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb')
    spaces={m:namespace(ROOT/f) for m,f in NAMES.items()}
    frame=base['load_benchmark']()
    checks={}
    for method,ns in spaces.items():
        nbformat.validate(nbformat.read(ROOT/NAMES[method],as_version=4))
        for key in ['MODEL_SNAPSHOT','EMBEDDING_MODEL','NRM_RETRY_ON_TRUNCATION','CSVQA_MODE_BY_ROUTE','CSV_SOLVER_INSTRUCTIONS']:
            assert ns[key]==base[key],(method,key)
        for fn in ['execute_code','_source_candidate','objective_is_correct','case_fields']:
            assert ns[fn].__code__.co_code==base[fn].__code__.co_code,(method,fn)
        # Parameters at defaults, not only the documented values.
        assert ns['objective_is_correct'].__kwdefaults__==base['objective_is_correct'].__kwdefaults__
        assert ns['BASE_NOTEBOOK_SHA256']==hashlib.sha256((ROOT/'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb').read_bytes()).hexdigest()
    for method in ['rag_only','few_shot_only']:
        ns=spaces[method]; f=ns['attach_cached_classifications'](frame)
        assert len(f)==101 and f['cached_predicted_label'].notna().all()
        assert int(f['cached_predicted_label'].eq(f['true_label']).sum())==95
        # A question change must not silently accept a stale classification.
        changed=frame.iloc[:1].copy();changed['Query']='Different question'
        try: ns['attach_cached_classifications'](changed)
        except ValueError: pass
        else: raise AssertionError('stale classification accepted')
        ns['invoke_classifier']=lambda *a:(_ for _ in ()).throw(AssertionError('Ablation classifier called'))
        def stop_after_route(route,query,address):
            assert route==f.iloc[0]['cached_assigned_route']
            raise RuntimeError('intentional formulation boundary')
        old=ns['invoke_formulation'];ns['invoke_formulation']=stop_after_route
        try:ns['execute_pipeline_case'](f.iloc[0].to_dict())
        except RuntimeError as e: assert str(e)=='intentional formulation boundary'
        ns['invoke_formulation']=old
        checks[method+'_cached_classification']=101
    rag=spaces['rag_only']
    assert rag['retrieve_rag_examples']('NRM','query',3)==[]
    assert rag['build_formulation_examples']('TP','query',3)==''
    assert rag['retrieve_csv_code_example']('RA','query')==''
    for fn in ['build_csvqa_components','_ask_extraction_plan','_execute_plan','csv_schema_preview']:
        assert rag[fn].__code__.co_code==base[fn].__code__.co_code,fn
    checks['rag_current_case_csvqa_and_planner_preserved']=True
    few=spaces['few_shot_only']
    assert 'build_csvqa_components' not in few and '_ask_extraction_plan' not in few and '_execute_plan' not in few
    assert few['_get_rag_store'].__wrapped__.__code__.co_code==base['_get_rag_store'].__wrapped__.__code__.co_code
    calls=[]
    class FakeLLM:
        def invoke(self,messages):
            calls.append(messages[0].content)
            return SimpleNamespace(content='m = None')
    saved_llm=few['make_llm']; few['make_llm']=lambda *a,**k:FakeLLM()
    saved_examples=few['retrieve_csv_code_example'];few['retrieve_csv_code_example']=lambda *a:'Retained code reference'
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'source.csv'; p.write_text('ID,value,empty\n001,1.234500,\n002,987654.321,\n003,-12.500,\n')
        payload=few['direct_source_payload'](str(p),'NRM')
        assert [r['values']['ID'] for r in payload['tables'][0]['records']]==['001','002','003']
        assert [r['values']['value'] for r in payload['tables'][0]['records']]==['1.234500','987654.321','-12.500']
        obs=json.dumps(payload,ensure_ascii=False)
        few['formulate_with_csvqa']('query',str(p),'NRM','unused','unused','base example','unused')
        assert '987654.321' in calls[-1] and '001' in calls[-1] and '003' in calls[-1]
        few['_generate_code']('Symbolic current model','NRM','query',data_payload=obs)
        assert obs not in calls[-1] and '987654.321' not in calls[-1] and '1.234500' not in calls[-1]
        assert 'Retained code reference' in calls[-1] and 'CSVQA_DATA' in calls[-1]
        few['_generate_code']('Numerical current model','RA','query',legacy_observation=obs)
        assert obs not in calls[-1] and '987654.321' not in calls[-1]
        # Execute generated NRM code receives data only from the existing pipeline's runtime binding.
        old_form=few['invoke_formulation'];old_code=few['get_csv_code'];old_exec=few['execute_code']
        few['invoke_formulation']=lambda *a:{'formulation':'symbolic','observation':obs,'trace':{'status':'PYTHON_FULL_CSV'}}
        few['get_csv_code']=lambda *a,**k:'m = None'
        def execute_injected(code):
            assert code.startswith('CSVQA_DATA = ') and '987654.321' in code
            return 1.0,[]
        few['execute_code']=execute_injected
        case=frame.iloc[0].to_dict();case.update(cached_predicted_label='NRM',cached_assigned_route='NRM')
        assert few['execute_pipeline_case'](case)['final_ok']
        few['invoke_formulation']=old_form;few['get_csv_code']=old_code;few['execute_code']=old_exec
    few['make_llm']=saved_llm;few['retrieve_csv_code_example']=saved_examples
    checks['fewshot_full_observation_model_only']=True
    loto=spaces['examples_and_route'];rows=[]
    for target in loto['CLASS_LABELS']:
        manifest=loto['set_loto_fold'](target)
        rf=loto['loto_reference_frame']()
        assert not rf['Type'].map(loto['normalize_problem_class']).eq(target).any()
        disabled=manifest['disabled_workflow_route']
        assert disabled not in manifest['allowed_routes']
        assert all(loto['class_to_workflow_route'](label)!=disabled for label in manifest['allowed_labels'])
        # Guard is checked before calling the base formulation function or retrieving examples.
        for fn,args in [('invoke_formulation',(disabled,'query','file.csv')),
                        ('retrieve_rag_examples',(disabled,'query',1)),
                        ('_generate_code',('model',disabled,'query'))]:
            try: loto[fn](*args)
            except loto['DisabledRouteError']: pass
            else: raise AssertionError((target,fn,'disabled route accepted'))
        for line in loto['prefix'].splitlines():
            if line.strip().startswith('Final Answer:'):
                assert line.split(':',1)[1].strip()!=target
        rows.append({'held_out_type':target,'removed_rows':[x['row_index'] for x in manifest['removed_examples']],
                     'disabled_route':disabled,'remaining':len(rf)})
    with contextlib.redirect_stdout(io.StringIO()): loto['loto_preflight']()
    variants=loto['load_variants_for_baseline'](ROOT/'benchmark_dataset/questions.csv')
    loto['validate_loto_cases'](variants)
    columns=loto['load_redundant_sheets_for_baseline'](ROOT/'redundancy_complete/redundant_instances.xlsx',loto['COLUMN_SHEETS'])
    for f in columns.values():loto['validate_loto_cases'](f)
    checks.update(loto_fold_boundaries=rows,variants_cases=len(variants),redundant_sheets={k:len(f) for k,f in columns.items()})
    out=ROOT/'outputs/ablation_loto_0927';out.mkdir(exist_ok=True)
    (out/'boundary_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
