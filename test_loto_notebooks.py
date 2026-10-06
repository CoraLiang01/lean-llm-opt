"""Offline LOTO regression checks using the real reference/benchmark CSVs.

Run: python test_loto_notebooks.py
Requires pandas and numpy, but no LangChain, model service, embeddings or Gurobi.
All results are written inside a temporary directory.
"""
import ast,contextlib,fcntl,hashlib,io,json,math,os,pprint,re,tempfile
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
NAMES=['LOTO_Examples_Only_GPT4.1_Large-scale.ipynb', 'LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb', 'LOTO_Examples_Only_gpt_oss_20b_Large-scale.ipynb', 'LOTO_Examples_And_Route_gpt_oss_20b_Large-scale.ipynb']
class Document:
 def __init__(self,page_content,metadata=None):self.page_content=page_content;self.metadata=metadata or {}
class Index:
 def __init__(self,documents):self.docs=documents;self.k=None
 def as_retriever(self,search_kwargs):self.k=search_kwargs['k'];return self
 def invoke(self,query):return self.docs[:self.k]
 def similarity_search(self,query,k=1,**kw):return self.docs[:k]
class FAISS:
 builds=[]
 @classmethod
 def from_documents(cls,documents,embeddings):
  assert documents,'Empty FAISS index';cls.builds.append(documents);return Index(documents)
class Tool:
 def __init__(self,**kw):self.__dict__.update(kw)
class DummyParser:
 def __init__(self,**kwargs):self.__dict__.update(kwargs)
class AgentAction(SimpleNamespace):
 pass
class AgentFinish(SimpleNamespace):
 pass
class OutputParserException(ValueError):
 def __init__(self,message,*,observation=None,**kwargs):
  super().__init__(message);self.observation=observation
class ReActParser:
 def parse(self,text):
  return AgentAction(tool='FileQA',tool_input='query',log=text)
class FakeAgent:
 def __init__(self,ns,kw):self.ns=ns;self.kw=kw
 def invoke(self,query):
  self.ns['_last_classifier_prefix']=self.kw['agent_kwargs']['prefix']
  obs=self.kw['tools'][0].func(query)
  return {'output':self.ns['_test_label'], 'intermediate_steps':[(SimpleNamespace(tool='FileQA'),obs)]}
def namespace(book,result_root):
 def forbidden(*a,**k):raise AssertionError('Unexpected real model/embedding dependency')
 ns=dict(ast=ast,contextlib=contextlib,fcntl=fcntl,hashlib=hashlib,io=io,json=json,math=math,os=os,pprint=pprint,re=re,tempfile=tempfile,
  lru_cache=lru_cache,Path=Path,RLock=RLock,Any=Any,quote=quote,np=np,pd=pd,
  Document=Document,FAISS=FAISS,Tool=Tool,AgentOutputParser=DummyParser,ReActSingleInputOutputParser=ReActParser,
  OutputParserException=OutputParserException,AgentAction=AgentAction,AgentFinish=AgentFinish,
  AgentType=SimpleNamespace(ZERO_SHOT_REACT_DESCRIPTION='react'),HumanMessage=lambda **kw:SimpleNamespace(**kw),
  ModelOutputTruncated=RuntimeError,PROJECT_ROOT=ROOT,RESULTS_DIR=result_root,BENCHMARK_PATH=ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101.csv',
  make_llm=forbidden,make_embeddings=lambda:object(),display=lambda *a:None,require_api_key=lambda:None)
 # Copy only literal configuration, never dotenv/import/service calls.
 for n in ast.parse(''.join(book['cells'][3]['source'])).body:
  if isinstance(n,ast.Assign):
   try:value=ast.literal_eval(n.value)
   except (ValueError,TypeError):continue
   for target in n.targets:
    if isinstance(target,ast.Name):ns[target.id]=value
 ns['NOTEBOOK_PATH']=ROOT/book['_filename']
 ns['initialize_agent']=lambda **kw:FakeAgent(ns,kw)
 ns['_model_prompts']=[]
 def model_invoke(messages):
  ns['_model_prompts'].extend(message.content for message in messages)
  return SimpleNamespace(content=ns.get('_fallback_label','RA'))
 ns['make_llm']=lambda:SimpleNamespace(invoke=model_invoke)
 for i in [7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]:
  exec(compile(''.join(book['cells'][i]['source']),f'{book["_filename"]}:cell{i}','exec'),ns)
 return ns
def check_classifier_prompts(ns,is_oss):
 allowed=ns['loto_allowed_labels']()
 actual=ns['_last_classifier_prefix']
 if ns['LOTO_REMOVE_ROUTE']:
  assert actual.startswith(ns['_LOTO_BASE_CLASSIFIER_PREFIX'])
  match=re.search(r'^The only available Final Answer labels are: (.+)\.$',actual,re.M)
  assert match and match.group(1).split(', ')==allowed,actual
 else:
  assert actual==ns['_LOTO_BASE_CLASSIFIER_PREFIX'],'Examples-only changed the classifier prefix'
 if not is_oss:return
 # Exercise the real tolerant parser's duplicate-tool correction, not just its source text.
 parser=ns['OllamaLabelTolerantReActOutputParser'](fallback_input='query',tool_seen=True)
 try:parser.parse('Thought: inspect data\nAction: FileQA\nAction Input: query')
 except OutputParserException as exc:
  match=re.search(r'Return exactly one of (.+) as the Final Answer\.',exc.observation)
  assert match and match.group(1).split(', ')==allowed,exc.observation
 else:raise AssertionError('Duplicate-tool correction was not exercised')
 # An invalid initial label must enter the real label-completion fallback.
 original_label=ns['_test_label'];ns['_test_label']='unrecognized output'
 ns['_fallback_label']=allowed[-1];ns['_model_prompts'].clear()
 try:
  with contextlib.redirect_stdout(io.StringIO()):result=ns['invoke_classifier']('a problem description only')
  assert result['normalized_label']==allowed[-1] and len(ns['_model_prompts'])==1
  prompt=ns['_model_prompts'][0]
  match=re.search(r'^return exactly one (?:AVAILABLE )?label token: (.+)\.$',prompt,re.M)
  assert match and match.group(1).replace(', or ',', ').split(', ')==allowed,prompt
 finally:ns['_test_label']=original_label
def test_loto_notebooks():
 books={name:json.loads((ROOT/name).read_text()) for name in NAMES}
 for index in (37,41,47):
  shared=''.join(books[NAMES[0]]['cells'][index]['source'])
  assert all(''.join(book['cells'][index]['source'])==shared for book in books.values()),f'LOTO control cell {index} diverged'
 with tempfile.TemporaryDirectory() as tmp:
  for name in NAMES:
   book=books[name];book['_filename']=name
   assert len({cell['id'] for cell in book['cells']}) == len(book['cells'])
   # Saved outputs/execution counters do not change the experiment protocol.
   assert len(book['cells'])==50
   for index in (47,49):
    assert {'experiment','loto-report'}.issubset(book['cells'][index].get('metadata',{}).get('tags',[]))
   def fingerprinted_source(cells):
    return '\n'.join(''.join(cell['source']) for cell in cells
     if cell['cell_type']=='code' and 'experiment' not in cell.get('metadata',{}).get('tags',[]))
   assert fingerprinted_source(book['cells'])==fingerprinted_source(book['cells'][:46]), 'Reporting changed cache-fingerprinted source'
   for i,c in enumerate(book['cells']):
    if c['cell_type']=='code':compile(''.join(c['source']),f'{name}:{i}','exec')
   ns=namespace(book,Path(tmp)/name.replace('.ipynb',''))
   remove_route='Examples_And_Route_' in name
   assert ns['LOTO_REMOVE_ROUTE'] is remove_route
   assert ns['LOTO_VARIANT']==('examples_and_route' if remove_route else 'examples_only')
   frame,preview=ns['loto_preflight']()
   assert preview['cases'].tolist()==[9,25,22,14,5,18,8],preview
   assert preview['examples_removed'].tolist()==[1,1,1,1,1,8,2]
   assert sum(preview['cases'])==101 and frame['problem_id'].nunique()==101
   hashes=[]
   for heldout in ns['CLASS_LABELS']:
    manifest=ns['set_loto_fold'](heldout)
    filtered=ns['loto_reference_frame']()
    labels=filtered.Type.map(ns['normalize_problem_class'])
    assert not labels.eq(heldout).any()
    if heldout=='Others':assert labels.eq('Mixture').sum()==8
    if heldout=='Mixture':assert labels.eq('Others').sum()==2
    disabled=ns['class_to_workflow_route'](heldout) if ns['LOTO_REMOVE_ROUTE'] else None
    assert manifest['disabled_workflow_route']==disabled
    if disabled=='Others':assert 'Mixture' not in ns['loto_allowed_labels']() and 'Others' not in ns['loto_allowed_labels']()
    # Exercise both classifier reference implementations; stale caches would retain the prior fold.
    docs=ns['_get_classifier_retriever']().docs
    removed_prompts={item['prompt'] for item in manifest['removed_examples']}
    assert len(docs)==manifest['reference_count_after']
    for doc in docs:
     assert not any(doc.page_content.startswith('prompt: '+prompt+'\n') for prompt in removed_prompts)
    assert len(ns['_get_classifier_retriever']().invoke('query'))==5
    if 'GPT4.1' in name:
     if heldout in ['TP','NRM','RA','FLP','AP']:
      builds=len(FAISS.builds)
      assert ns['retrieve_rag_examples'](heldout,'query',k=3)==[]
      assert ns['build_formulation_examples'](heldout,'query',k=3)==''
      assert ns['retrieve_csv_code_example'](heldout,'query')==''
      assert len(FAISS.builds)==builds
     if heldout in ['Others','Mixture']:
      assert len(ns['load_rag_examples']('Others'))==(8 if heldout=='Others' else 2)
    else:
     # Others formatting calls as_retriever through this actual entry point.
     empty=ns['_EmptyExampleIndex']()
     assert ns['retrieve_examples'](empty,'query',k=3)==[]
     assert ns['build_few_shot_Other'](empty,'query',k=3,t='Code')==''
     if heldout in ['TP','NRM','RA','FLP','AP']:
      builds=len(FAISS.builds);assert ns['get_route_retriever'](heldout,3).invoke('query')==[]
      assert len(FAISS.builds)==builds
     if heldout in ['Others','Mixture']:
      assert len(ns['get_others_store']().docs)==(8 if heldout=='Others' else 2)
    ns['_test_label']=ns['loto_allowed_labels']()[0]
    with contextlib.redirect_stdout(io.StringIO()):classification=ns['invoke_classifier']('a problem description only')
    assert classification['normalized_label']==ns['_test_label']
    check_classifier_prompts(ns,'gpt_oss_20b' in name)
    # The dispatcher must reject a disabled route before touching formulation/data/code.
    visits=[]
    ns['_LOTO_BASE_INVOKE_FORMULATION']=lambda *a:visits.append(a) or {'formulation':'model'}
    if disabled:
     try:ns['invoke_formulation'](disabled,'q','some.csv')
     except ns['DisabledRouteError']:pass
     else:raise AssertionError('Disabled route executed')
     assert not visits
    for route in manifest['allowed_routes']:
     ns['invoke_formulation'](route,'q','some.csv')
    assert len(visits)==len(manifest['allowed_routes'])
    first=frame.iloc[0].to_dict();first['dataset_address']=''
    hashes.append(ns['case_fingerprint'](first,source_fingerprint='same-source'))
   assert len(set(hashes))==7,'Fingerprint lacks semantic fold isolation'
   # Actual original pipelines + guarded dispatch; no real formulation or solver used.
   for heldout in ns['CLASS_LABELS']:
    ns['set_loto_fold'](heldout)
    case=frame.loc[frame.true_label.eq(heldout)].iloc[0].to_dict()
    ns['_test_label']=ns['loto_allowed_labels']()[0] if ns['LOTO_REMOVE_ROUTE'] else heldout
    ns['_LOTO_BASE_INVOKE_FORMULATION']=lambda *a:{'formulation':'valid model', 'code':'m = None', 'observation':json.dumps({'validation':{'status':'OK'},'tables':[]}), 'trace':{'status':'OK'}}
    ns['execute_code']=lambda code:(case['label_objective'],[])
    with contextlib.redirect_stdout(io.StringIO()):record=ns['execute_pipeline_case'](case)
    assert record['final_ok'] and record['held_out_type']==heldout
    if ns['LOTO_REMOVE_ROUTE']:
     ns['_test_label']=heldout
     try:
      with contextlib.redirect_stdout(io.StringIO()):ns['execute_pipeline_case'](case)
     except ns['DisabledRouteError'] as exc:
      assert exc.pipeline_context['pipeline_stage']=='route_selection'
      assert exc.pipeline_context['predicted_label']==heldout
     else:raise AssertionError('Original pipeline bypassed guard')
   # Full runner exercises save/materialize/fingerprint-based resume and stage reports.
   raw_run=ns['execute_pipeline_case'];run_calls=[]
   def pipeline_stub(case,forced_route=None):
    run_calls.append(case['problem_id'])
    state=ns['require_loto_fold']();label=state['allowed_labels'][0] if ns['LOTO_REMOVE_ROUTE'] else state['held_out_type']
    result={**ns['case_fields'](case),**ns['loto_record_fields'](),'predicted_label':label,'assigned_route':ns['class_to_workflow_route'](label),
     'forced_route':None,'experiment_mode':'automatic','generated_model':'model','solve_code':'m = None','csvqa_observation':'table',
     'final_solution':[],'final_objective':case['label_objective'],'final_ok':True,'record_status':'completed','cache_source':'computed','pipeline_stage':'completed','route_allowed':True}
    return result
   ns['execute_pipeline_case']=pipeline_stub;ns['_source_fingerprint']=lambda:'source-v1'
   selected=[int(frame.index[frame.true_label.eq(t)][0]) for t in ns['CLASS_LABELS']]
   with contextlib.redirect_stdout(io.StringIO()):records=ns['run_loto'](rows=selected)
   assert len(records)==7 and len(run_calls)==7
   assert all(r['classification_correct'] is None for r in records) if ns['LOTO_REMOVE_ROUTE'] else all(r['classification_correct'] for r in records)
   with contextlib.redirect_stdout(io.StringIO()):cached=ns['run_loto'](rows=selected)
   assert len(run_calls)==7 and all(r['cache_source']=='csv' for r in cached)
   ns['_source_fingerprint']=lambda:'source-v2'
   with contextlib.redirect_stdout(io.StringIO()):ns['run_loto'](rows=selected)
   assert len(run_calls)==14,'Stale results were reused'
   # Failures are retried even on oss; they never become matching-success cache entries.
   def fail(case,forced_route=None):
    run_calls.append('FAIL');raise RuntimeError('mock failure')
   ns['execute_pipeline_case']=fail;ns['_source_fingerprint']=lambda:'source-v3'
   with contextlib.redirect_stdout(io.StringIO()):failed=ns['run_loto'](folds=['RA'],rows=selected)
   assert len(failed)==1 and not failed[0]['final_ok']
   ns['execute_pipeline_case']=pipeline_stub
   with contextlib.redirect_stdout(io.StringIO()):retried=ns['run_loto'](folds=['RA'],rows=selected)
   assert retried[0]['final_ok'] and retried[0]['cache_source']=='computed'
   # Completion means a successful solve; an incorrect objective is still reusable.
   def wrong_objective(case,forced_route=None):
    result=pipeline_stub(case,forced_route)
    result['final_objective']=case['label_objective']+max(abs(case['label_objective']),1.0)
    return result
   ns['execute_pipeline_case']=wrong_objective;ns['_source_fingerprint']=lambda:'source-v4'
   with contextlib.redirect_stdout(io.StringIO()):wrong=ns['run_loto'](folds=['RA'],rows=selected)
   assert wrong[0]['record_status']=='completed' and wrong[0]['final_ok'] and not wrong[0]['solution_correct']
   calls_before=len(run_calls)
   with contextlib.redirect_stdout(io.StringIO()):wrong_cached=ns['run_loto'](folds=['RA'],rows=selected)
   assert len(run_calls)==calls_before and wrong_cached[0]['cache_source']=='csv'
   assert wrong_cached[0]['final_ok'] and not wrong_cached[0]['solution_correct']
   assert wrong_cached[0]['final_objective']==wrong[0]['final_objective']
   print('PASS',name,': 7 folds; shared controls/config; exact CSV filtering; empty examples; dynamic classifier prompts; route guard; native pipeline; correct/incorrect-objective cache/retry/report integration')
 for name in NAMES:
  book=json.loads((ROOT/name).read_text())
  config={}
  for node in ast.parse(''.join(book['cells'][3]['source'])).body:
   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
    try:config[node.targets[0].id]=ast.literal_eval(node.value)
    except (ValueError,TypeError):pass
  assert hashlib.sha256((ROOT/config['LOTO_BASE_NOTEBOOK']).read_bytes()).hexdigest()==config['LOTO_BASE_SHA256'], 'Baseline snapshot changed'
 print('PASS: baseline snapshots unchanged; fake LLM/FAISS/solver only, no real API calls.')


def test_loto_saved_report():
 """Exercise the independent reporting cell without loading any experiment services."""
 book=json.loads((ROOT/NAMES[0]).read_text())
 ns={}
 exec(compile(''.join(book['cells'][47]['source']),'loto-report-cell','exec'),ns)
 report=ns['loto_saved_report']
 labels=['TP','NRM','RA','FLP','AP','Mixture','Others']
 routes=['TP','NRM','RA','FLP','AP','Others']
 route=lambda label:'Others' if label in {'Mixture','Others'} else label
 benchmark_rows=[]
 for label,count in zip(labels,[9,25,22,14,5,18,8]):
  for _ in range(count):
   pid=f'OR-{len(benchmark_rows)+1:03d}'
   benchmark_rows.append({'problem_id':pid,'Query':f'Query {pid}',
    'Problem Type':label,'Label-objective':0.0 if pid=='OR-001' else 100.0})
 expected={row['problem_id']:row for row in benchmark_rows}
 def record(pid,variant='examples_only',**overrides):
  case=expected[pid];gold=case['Problem Type']
  result={'problem_id':pid,'held_out_type':gold,'true_label':gold,'loto_variant':variant,
   'predicted_label':gold,'assigned_route':route(gold),'final_ok':'True',
   'final_objective':str(case['Label-objective']),'label_objective':str(case['Label-objective']),
   'solution_correct':'True','pipeline_stage':'completed','execution_error_type':''}
  result.update(overrides)
  return result
 def write_records(folder,records):
  for label in labels:
   selected=[row for row in records if row['held_out_type']==label]
   if selected:
    path=folder/label/'results.csv';path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(selected).to_csv(path,index=False)
 def call(folder,benchmark_path,variant='examples_only'):
  return report(folder,benchmark_path,variant=variant,show=False)
 def assert_matrix_totals(tables):
  expected_counts=pd.Series(dict(zip(labels,[9,25,22,14,5,18,8])))
  for name,choices in [('classification_matrix',labels),('route_matrix',routes)]:
   matrix=tables[name]
   assert matrix.index.tolist()==labels
   assert matrix.columns.tolist()==choices+['Missing','Invalid','Not run']
   assert matrix.to_numpy().sum()==101
   assert matrix.sum(axis=1).to_dict()==expected_counts.to_dict()
 with tempfile.TemporaryDirectory() as tmp:
  folder=Path(tmp);benchmark_path=folder/'benchmark.csv'
  benchmark_fixture=pd.DataFrame(benchmark_rows)
  benchmark_fixture.loc[99,'Problem Type']='Sales-Based Linear Programming'
  benchmark_fixture.loc[100,'Problem Type']='Others -Knapsack'
  benchmark_fixture.to_csv(benchmark_path,index=False)
  # Real benchmark compatibility must not depend on the simplified mock labels/objectives.
  real_empty=report(folder/'unrun_real_benchmark',ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101.csv',
   variant='examples_only',output_dir=folder/'real_benchmark_reports',show=False)
  real_total=real_empty['overall'].iloc[0]
  assert real_total['expected_cases']==101 and real_total['recorded_cases']==0 and real_total['pending_cases']==101
  assert real_total['failed']==0 and not real_total['complete_101']
  assert real_empty['case_details']['true_label'].isin(labels).all()
  assert real_empty['case_details']['label_objective'].notna().all()
  assert_matrix_totals(real_empty)
  # An unrun oss experiment has no false failures and no divide-by-zero rates.
  for variant in ['examples_only','examples_and_route']:
   empty=call(folder/('empty_oss_'+variant),benchmark_path,variant)
   total=empty['overall'].iloc[0]
   assert total['expected_cases']==101 and total['recorded_cases']==0 and total['pending_cases']==101
   assert total['failed']==0 and total['solved']==0 and not total['complete_101']
   for field in ['solve_rate','objective_accuracy','accuracy_among_solved','classification_accuracy']:
    assert pd.isna(total[field]),field
   if variant=='examples_and_route':assert pd.isna(total['classification_correct'])
   assert not empty['case_details']['has_record'].any()
   assert empty['case_details'][['final_ok','solution_correct','classification_correct']].isna().all().all()
   assert empty['failures'].empty
   assert_matrix_totals(empty)
   assert empty['classification_matrix']['Not run'].sum()==101
   assert empty['route_matrix']['Not run'].sum()==101
  # Match the benchmark's monetary and prefixed scientific objective formats.
  numeric_rows=[dict(row) for row in benchmark_rows]
  for row,value in zip(numeric_rows,['$0.00','$1,234.50','Optimal objective 1.2e3']):
   row['Label-objective']=value
  numeric_benchmark=folder/'benchmark_formatted_objectives.csv'
  pd.DataFrame(numeric_rows).to_csv(numeric_benchmark,index=False)
  numeric_dir=folder/'formatted_objectives'
  write_records(numeric_dir,[
   record('OR-001',final_objective='0',label_objective='$0.00'),
   record('OR-002',final_objective='1234.5',label_objective='$1,234.50'),
   record('OR-003',final_objective='1.2e3',label_objective='Optimal objective 1.2e3'),
  ])
  numeric_tables=call(numeric_dir,numeric_benchmark)
  numeric_details=numeric_tables['case_details'].set_index('problem_id').loc[['OR-001','OR-002','OR-003']]
  assert numeric_details['label_objective'].tolist()==[0.0,1234.5,1200.0]
  assert numeric_details['final_objective'].tolist()==[0.0,1234.5,1200.0]
  assert numeric_details['solution_correct'].all()
  assert not numeric_details['benchmark_objective_changed'].any()
  assert not numeric_details['stored_match_disagrees'].any()
  assert numeric_tables['overall'].iloc[0]['objective_correct']==3
  partial_dir=folder/'partial'
  partial=[record('OR-001'),
   record('OR-002',final_ok='False',predicted_label='',assigned_route='',pipeline_stage='classification',execution_error_type='RuntimeError'),
   record('OR-003',final_ok='False',final_objective='',solution_correct='False',predicted_label='Unknown',assigned_route='Unknown',pipeline_stage='classification',execution_error_type='RuntimeError'),
   record('OR-004',final_objective='200'),
   record('OR-010',final_ok='1',solution_correct='1',final_objective='250',label_objective='250'),
   record('OR-035',final_ok='False',solution_correct='False',pipeline_stage='solve',execution_error_type='RuntimeError'),
   record('OR-076',predicted_label='Others',assigned_route='Others')]
  write_records(partial_dir,partial)
  # Stale top-level reports and a current one-row in-memory result must not be used.
  pd.DataFrame(partial[:1]).to_csv(partial_dir/'case_summary.csv',index=False)
  ns['loto_results']=partial[:1];ns['FOLDS']=['TP'];ns['ROWS']=[0]
  (partial_dir/'TP'/'fold_manifest.json').write_text('{"unchanged": true}')
  before={p:p.read_bytes() for p in partial_dir.rglob('*') if p.is_file()}
  tables=call(partial_dir,benchmark_path)
  assert all(p.read_bytes()==content for p,content in before.items()),'Reporting rewrote original results'
  total=tables['overall'].iloc[0]
  for field,value in {'recorded_cases':7,'pending_cases':94,'classification_correct':4,
   'solved':4,'failed':3,'objective_correct':3,'solved_not_matched':1,
   'stored_match_disagreements':2,'benchmark_objective_changes':1}.items():
   assert total[field]==value,(field,total[field],value)
  assert total['classification_accuracy']==4/7 and total['solve_rate']==4/7
  assert total['objective_accuracy']==3/7 and total['accuracy_among_solved']==3/4
  details=tables['case_details'].set_index('problem_id')
  assert details.loc['OR-001','final_ok'] and details.loc['OR-001','solution_correct']
  assert details.loc['OR-001','final_objective']==0
  assert not details.loc['OR-002','final_ok'] and not details.loc['OR-002','solution_correct']
  assert details.loc['OR-004','outcome']=='Solved, objective mismatch'
  assert details.loc['OR-010','benchmark_objective_changed'] and details.loc['OR-010','solution_correct']
  assert pd.isna(details.loc['OR-005','final_ok']) and details.loc['OR-005','outcome']=='Not run'
  assert tables['failures']['cases'].sum()==3
  assert_matrix_totals(tables)
  for matrix in ['classification_matrix','route_matrix']:
   assert tables[matrix].loc['TP','Missing']==1 and tables[matrix].loc['TP','Invalid']==1
   assert tables[matrix]['Not run'].sum()==94
  assert tables['classification_matrix'].loc['Mixture','Others']==1
  assert tables['route_matrix'].loc['Mixture','Others']==1
  assert all((partial_dir/'reports'/f'{name}.csv').is_file() for name in tables)
  # Simulate a full run followed by one re-run: all 101 fold rows still count.
  full_dir=folder/'full';complete=[record(pid) for pid in expected]
  write_records(full_dir,complete)
  tables=call(full_dir,benchmark_path)
  assert tables['overall'].iloc[0]['recorded_cases']==101 and tables['overall'].iloc[0]['objective_correct']==101
  changed=record('OR-035',final_objective='200')
  complete=[changed if row['problem_id']=='OR-035' else row for row in complete]
  write_records(full_dir,complete)
  pd.DataFrame([changed]).to_csv(full_dir/'case_summary.csv',index=False)
  ns['loto_results']=[changed];ns['FOLDS']=['RA'];ns['ROWS']=[34]
  tables=call(full_dir,benchmark_path)
  total=tables['overall'].iloc[0]
  assert total['complete_101'] and total['recorded_cases']==101 and total['pending_cases']==0
  assert total['solved']==101 and total['objective_correct']==100 and total['solved_not_matched']==1
  assert tables['by_type']['recorded_cases'].tolist()==[9,25,22,14,5,18,8]
  assert_matrix_totals(tables)
  # Experiment 2 keeps the matrices but never reports classification accuracy.
  removed_dir=folder/'route_removed';removed=[]
  for pid,case in expected.items():
   alternative='Others' if case['Problem Type']=='RA' else 'RA'
   removed.append(record(pid,'examples_and_route',predicted_label=alternative,assigned_route=alternative))
  removed[0].update(predicted_label='TP',assigned_route='TP')
  write_records(removed_dir,removed)
  alias_benchmark=folder/'benchmark_aliases.csv'
  pd.DataFrame(benchmark_rows).rename(columns={'Problem Type':'true_label','Label-objective':'label_objective'}).to_csv(alias_benchmark,index=False)
  tables=call(removed_dir,alias_benchmark,'examples_and_route')
  assert tables['overall'][['classification_correct','classification_accuracy']].isna().all().all()
  assert tables['by_type'][['classification_correct','classification_accuracy']].isna().all().all()
  assert tables['case_details']['classification_correct'].isna().all()
  assert tables['overall'].iloc[0]['forbidden_route_selections']==1
  assert tables['classification_matrix'].loc['TP','TP']==1
  assert_matrix_totals(tables)
  # Invalid saved populations must fail rather than silently deduplicate or regroup.
  invalid_cases=[
   ('duplicate',[record('OR-001'),record('OR-001')],'Duplicate problem ID'),
   ('unknown',[record('OR-001',problem_id='OR-999')],'unknown problem ID'),
   ('wrong_true_label',[record('OR-001',true_label='NRM')],'does not match fold'),
   ('wrong_benchmark_fold',[record('OR-010',held_out_type='TP',true_label='TP')],'does not match the current benchmark'),
   ('wrong_variant',[record('OR-001','examples_and_route')],'saved variant'),
  ]
  for name,records,message in invalid_cases:
   invalid_dir=folder/name;write_records(invalid_dir,records)
   try:call(invalid_dir,benchmark_path)
   except ValueError as exc:assert message in str(exc),(name,str(exc))
   else:raise AssertionError(f'{name} results were accepted')
 print('PASS: independent saved reports; empty/partial/full/re-run populations; string booleans and zero objectives; missing/invalid matrices; experiment-2 N/A; original-result preservation; duplicate/fold/variant rejection.')


if __name__ == '__main__':
 test_loto_notebooks()
 test_loto_saved_report()
