"""One full pass of the three final methods derived after full-model targets passed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from check_0927_experiments import NAMES, ROOT
from evaluate_0927_optimization import namespace

OUT=ROOT/'outputs/optimization_0927_20261006'
NS={}
CASE_DIR=None
CASE_DEADLINE=1800

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def init_worker():
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS','1')


def worker(item):
    global CASE_DIR
    method=item['method'];folder=Path(item['attempt_dir']);CASE_DIR=folder
    (folder/'attempt.json').write_text(json.dumps({'state':'running','problem_id':item['case']['problem_id']}))
    started=time.monotonic()
    timed_out=threading.Event()
    def deadline():
        timed_out.set();os.kill(os.getpid(),signal.SIGINT)
    timer=threading.Timer(CASE_DEADLINE,deadline);timer.daemon=True;timer.start()
    with (folder/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            if method not in NS:
                ns=namespace(Path(item['notebook']))
                class PromptRecorder(ns['BaseCallbackHandler']):
                    def on_chat_model_start(self,serialized,messages,**kwargs):
                        if CASE_DIR is None:return
                        data={'utc':datetime.now(timezone.utc).isoformat(),
                              'messages':[[{'role':m.type,'content':m.content} for m in group] for group in messages]}
                        with (CASE_DIR/'prompts.jsonl').open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
                original=ns['make_llm']
                def logged_llm(*a,**kw):
                    llm=original(*a,**kw)
                    llm.callbacks=list(llm.callbacks or [])+[PromptRecorder()]
                    return llm
                ns['make_llm']=logged_llm
                NS[method]=ns
            ns=NS[method];case=item['case']
            if method=='examples_and_route':
                state=ns.get('LOTO_STATE')
                if state is None or state['held_out_type']!=case['true_label']:
                    ns['set_loto_fold'](case['true_label'])
                (folder/'fold_manifest.json').write_text(json.dumps(ns['loto_manifest'](),ensure_ascii=False,indent=2))
            record=ns['execute_pipeline_case'](case)
        except BaseException as exc:
            ns=NS.get(method)
            traceback.print_exc()
            if timed_out.is_set():
                old=exc;exc=TimeoutError(f'External evaluation deadline {CASE_DEADLINE}s exceeded')
                exc.pipeline_context=getattr(old,'pipeline_context',{})
            if ns is None:
                raise
            record=ns['error_record'](item['case'],exc)
            if method=='examples_and_route':record.update(ns['loto_record_fields']())
        finally:
            timer.cancel()
        record=ns['finalize_record'](record,rel_tol=1e-4,abs_tol=1e-4)
        record.update(experiment_method=method,seconds=round(time.monotonic()-started,3),
            classification_source='final_full_v1_cache' if method!='examples_and_route' else 'fold_agent',
            classification_cache_sha256=item['classification_sha'],
            base_notebook_sha256=item['base_sha'],notebook_source_sha256=item['notebook_sha'],
            cache_fingerprint=ns['case_fingerprint'](item['case'],source_fingerprint=item['source_fingerprint']),
            evaluated_utc=datetime.now(timezone.utc).isoformat(),
            external_deadline_seconds=CASE_DEADLINE,external_deadline_exceeded=timed_out.is_set())
        (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str))
        (folder/'attempt.json').write_text(json.dumps({'state':'finished','problem_id':item['case']['problem_id']}))
    return record


def report(spaces,method):
    ns=spaces[method];root=OUT/(method+'_final')
    csv=root/'automatic/results.csv'
    records=ns['load_records'](csv)
    if not records:return
    with contextlib.redirect_stdout(io.StringIO()):
        ns['report_results'](records,root/'automatic')
        if method=='examples_and_route':ns['loto_report'](records,root/'automatic')
    summary={'recorded':len(records),'expected':101,
             'solved':sum(r.get('final_ok') is True for r in records),
             'objective_match':sum(r.get('solution_correct') is True for r in records),
             'classification_correct':sum(r.get('classification_correct') is True for r in records),
             'errors':sum(r.get('record_status')=='error' for r in records)}
    (root/'progress.json').write_text(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8)
    p.add_argument('--methods',nargs='+',choices=NAMES,default=list(NAMES))
    p.add_argument('--collect-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);spaces={};items=[]
    base_path=ROOT/'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'
    classification=ROOT/'outputs/optimization_0927_20261006/full_v1/classification_main.csv'
    baseline=namespace(base_path);frame=baseline['load_benchmark']()
    if len(frame)!=101:raise ValueError('Expected the 101-case baseline')
    input_hashes={str(base_path):sha(base_path),str(classification):sha(classification),
                  str(baseline['BENCHMARK_PATH']):sha(baseline['BENCHMARK_PATH'])}
    for path in (ROOT/'Large_Scale_Or_Files').rglob('*.csv'):input_hashes[str(path)]=sha(path)
    for address in frame['dataset_address']:
        for raw in address.splitlines():input_hashes[raw]=sha(raw)
    for method in args.methods:
        root=OUT/(method+'_final');root.mkdir(parents=True,exist_ok=True)
        frozen=root/'frozen_notebook.ipynb';original=ROOT/NAMES[method]
        if frozen.exists() and sha(frozen)!=sha(original):raise ValueError('Experiment code changed: use a new version')
        if not frozen.exists():frozen.write_bytes(original.read_bytes())
        ns=namespace(frozen);spaces[method]=ns;ns['require_api_key']()
        cases=frame.copy()
        if method!='examples_and_route':cases=ns['attach_cached_classifications'](cases)
        else:
            ns['loto_preflight']()
            # Schedule folds together to reuse only same-fold reference indexes.
            cases['fold_order']=cases['true_label'].map({v:i for i,v in enumerate(ns['CLASS_LABELS'])})
            cases=cases.sort_values('fold_order',kind='stable').drop(columns='fold_order')
        source=ns['_source_fingerprint']()
        manifest={'method':method,'baseline_notebook':str(base_path),'base_sha256':sha(base_path),
            'frozen_notebook':str(frozen),'notebook_sha256':sha(frozen),'source_fingerprint':source,
            'classification_cache':str(classification),'classification_sha256':sha(classification),
            'classification_policy':'reuse labels and routes only' if method!='examples_and_route' else 'fresh agent; gold type chooses fold only',
            'model':ns['MODEL_SNAPSHOT'],'temperature':0,'top_p':1,'n':1,'sdk_max_retries':ns['make_llm']().max_retries,
            'nrm_retry_on_truncation':ns['NRM_RETRY_ON_TRUNCATION'],'rel_tol':1e-4,'abs_tol':1e-4,
            'solver_instructions':ns['CSV_SOLVER_INSTRUCTIONS'],'input_sha256':input_hashes,
            'expected_cases':101,'selection':'all cases, one implementation, no case reruns',
            'external_case_deadline_seconds':CASE_DEADLINE,'python':sys.version,'executable':sys.executable}
        path=root/'run_manifest.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['input_sha256']!=manifest['input_sha256']:raise ValueError('Inputs changed since the frozen run')
        else:path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        existing={r['problem_id']:r for r in ns['load_records'](root/'automatic/results.csv')}
        for case in cases.to_dict('records'):
            if case['problem_id'] in existing:continue
            folder=root/'attempts'/case['problem_id'];folder.mkdir(parents=True,exist_ok=True)
            completed=folder/'result.json';attempt=folder/'attempt.json'
            if completed.exists():
                record=json.loads(completed.read_text());ns['save_record'](root/'automatic/results.csv',record);continue
            if args.collect_only:continue
            if attempt.exists():raise ValueError(f'Unfinished prior attempt; do not silently rerun: {folder}')
            items.append({'method':method,'case':case,'attempt_dir':str(folder),'notebook':str(frozen),
                'base_sha':sha(base_path),'notebook_sha':sha(frozen),'classification_sha':sha(classification),
                'source_fingerprint':source})
        report(spaces,method)
    if args.collect_only:return
    print(f'Running {len(items)} new recorded cases across {args.methods}; workers={args.workers}',flush=True)
    # Interleave methods so a configuration failure is detected early in all three.
    groups={m:[x for x in items if x['method']==m] for m in args.methods}
    items=[groups[m][i] for i in range(max([len(g) for g in groups.values()] or [0])) for m in groups if i<len(groups[m])]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        futures={pool.submit(worker,item):item for item in items}
        for completed,f in enumerate(as_completed(futures),1):
            item=futures[f];record=f.result();method=item['method']
            spaces[method]['save_record'](OUT/(method+'_final')/'automatic/results.csv',record)
            report(spaces,method)
            print(f"[{completed}/{len(items)}] {method} {record['problem_id']} route={record.get('assigned_route')} "
                  f"solved={record.get('final_ok')} objective_match={record.get('solution_correct')} "
                  f"seconds={record['seconds']} error={record.get('execution_error_type','')}",flush=True)
            if record.get('execution_error_type') in {'AuthenticationError','PermissionDeniedError'} or 'insufficient_quota' in record.get('execution_error',''):
                for other in futures:other.cancel()
                raise RuntimeError('Global API service failure; stop scheduling further cases')
    for method in args.methods:report(spaces,method)
    print('All recorded cases finished.',flush=True)

if __name__=='__main__':main()
