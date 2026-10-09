"""Evaluate 101 x six full-model routes, reusing only identical automatic-route cases."""
import argparse
import copy
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import evaluate_0927_optimization as engine

ROOT = engine.ROOT
BASE = ROOT / 'outputs/optimization_0927_20261006/full_review_v6_completed'
OUT = ROOT / 'outputs/optimization_0927_20261006/full_606_routes_v6_r1'

def worker(item):
    original = engine.NS['execute_pipeline_case']
    engine.NS['execute_pipeline_case'] = lambda case: original(case, forced_route=item['route'])
    try:
        r = engine.worker(item)
        r.update(forced_route=item['route'], experiment_mode='forced',
                 route_result_source='new_forced_execution', reused=False)
        (Path(item['folder'])/'result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str))
        return r
    finally:
        engine.NS['execute_pipeline_case'] = original

def progress(ns):
    rows=ns['load_records'](OUT/'forced/results.csv')
    data={'expected':606,'recorded':len(rows),'reused':sum(r.get('route_result_source')=='reused_full_automatic' for r in rows),
          'newly_executed':sum(r.get('route_result_source')=='new_forced_execution' for r in rows),
          'by_route':{}}
    for route in ns['WORKFLOW_ROUTES']:
        g=[r for r in rows if r.get('forced_route')==route]
        data['by_route'][route]={'expected':101,'recorded':len(g),'objective_match':sum(r.get('solution_correct') is True for r in g),
                                 'solved':sum(r.get('final_ok') is True for r in g),'errors':sum(r.get('record_status')=='error' for r in g)}
    (OUT/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return rows,data

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--collect-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    status_path = OUT/'run_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'completed':
        print('Frozen 606-route run is already complete; see '+str(OUT/'report/report.md'),flush=True)
        return
    raw=(BASE/'frozen_notebook.ipynb').read_bytes();frozen=OUT/'frozen_notebook.ipynb'
    if frozen.exists() and frozen.read_bytes()!=raw:raise ValueError('Frozen source changed')
    if not frozen.exists():frozen.write_bytes(raw)
    ns=engine.namespace(frozen);frame=ns['load_benchmark']();ns['require_api_key']()
    if len(frame)!=101:raise ValueError('Expected 101 cases')
    base_manifest=json.loads((BASE/'run_manifest.json').read_text());hashes={}
    paths={str(ns['BENCHMARK_PATH']),str(ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv')}
    for address in frame['dataset_address']:paths.update(address.splitlines())
    for path in sorted(paths):
        value=engine.sha(path)
        if base_manifest['input_sha256'].get(path)!=value:raise ValueError('Inputs differ from reusable full results: '+path)
        hashes[path]=value
    manifest={'expected':606,'routes':ns['WORKFLOW_ROUTES'],'model':ns['MODEL_SNAPSHOT'],'embedding_model':ns['EMBEDDING_MODEL'],
              'notebook_sha256':engine.sha(frozen),'input_sha256':hashes,'rel_tol':1e-4,'abs_tol':1e-4,'external_deadline_seconds':engine.DEADLINE,
              'sdk_max_retries':1,'reuse_source':str(BASE),'selection':'All 101 x six routes; reuse same-route automatic outcomes, including mismatches; no outcome-based selection',
              'route_policy':'Forced route predetermined by complete Cartesian product, not by gold label or reference answer'}
    mp=OUT/'run_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Run manifest changed')
    if not mp.exists():mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for f in OUT.glob('attempts/*/*/result.json'):ns['save_record'](OUT/'forced/results.csv',json.loads(f.read_text()))
    rows,_=progress(ns);existing={(r['problem_id'],r['forced_route']) for r in rows};cases=frame.to_dict('records')
    with (BASE/'automatic/results.csv').open(encoding='utf-8-sig') as f:automatic={r['problem_id']:r for r in csv.DictReader(f) if r['benchmark']=='main'}
    for case in cases:
        old=automatic[case['problem_id']];route=old['assigned_route'];key=(case['problem_id'],route)
        if old['query']!=str(case['Query']) or ns['normalize_data_address'](old['dataset_address'])!=ns['normalize_data_address'](case['dataset_address']):raise ValueError('Case identity changed')
        if key in existing:continue
        origin=BASE/'attempts'/case['problem_id']/'result.json';r=copy.deepcopy(json.loads(origin.read_text()))
        if r['notebook_source_sha256']!=manifest['notebook_sha256']:raise ValueError('Reusable source mismatch')
        r.update(forced_route=route,experiment_mode='forced',route_result_source='reused_full_automatic',reused=True,
                 reuse_source_result=str(origin),reuse_source_seconds=r.get('seconds'),seconds=0,cache_source='reused_same_full_route')
        ns['save_record'](OUT/'forced/results.csv',r);existing.add(key)
    rows,data=progress(ns)
    if args.collect_only:return
    if any(engine.credit_exhausted(r) for r in rows):raise ValueError('Retain quota failures; use a separate restored-credit round')
    items=[]
    for case in cases:
        for route in ns['WORKFLOW_ROUTES']:
            if (case['problem_id'],route) in existing:continue
            folder=OUT/'attempts'/route/case['problem_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'attempt.json').exists():raise ValueError('Unfinished attempt requires explicit accounting: '+str(folder))
            items.append({'case':case,'route':route,'folder':str(folder),'benchmark':'main','notebook_sha':manifest['notebook_sha256']})
    print(f'Reused {data["reused"]}; running {len(items)} missing route combinations; workers={args.workers}',flush=True)
    blocked=False
    with ProcessPoolExecutor(max_workers=args.workers,initializer=engine.initialize,initargs=(str(frozen),'full')) as pool:
        futures={pool.submit(worker,item):item for item in items}
        for i,future in enumerate(as_completed(futures),1):
            r=future.result();ns['save_record'](OUT/'forced/results.csv',r);_,data=progress(ns)
            print(f'[{i}/{len(items)}] {r["problem_id"]} {r["forced_route"]} objective={r.get("solution_correct")} solved={r.get("final_ok")} seconds={r["seconds"]} error={r.get("execution_error_type","")}',flush=True)
            if engine.credit_exhausted(r) or r.get('execution_error_type') in {'AuthenticationError','PermissionDeniedError'}:
                blocked=True
                for f in futures:f.cancel()
                print('API service blocked; preserving in-flight results.',flush=True);break
    for f in OUT.glob('attempts/*/*/result.json'):ns['save_record'](OUT/'forced/results.csv',json.loads(f.read_text()))
    rows,data=progress(ns)
    status={'state':'externally_blocked' if blocked else 'completed' if len(rows)==606 else 'incomplete','progress':data}
    (OUT/'run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    if blocked:raise SystemExit(3)
    ns['report_results'](rows,OUT/'forced')
    print('All 606 route combinations recorded.',flush=True)

if __name__=='__main__':main()
