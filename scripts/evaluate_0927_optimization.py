"""Versioned, single-pass evaluations of the minimally revised 0927 notebooks."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/optimization_0927_20261006'
NS = None
CASE_DIR = None
DEADLINE = 1800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def credit_exhausted(record):
    return (record.get('execution_error_type') == 'RateLimitError'
            and 'credit_balance_exhausted' in str(record.get('execution_error', '')))


def namespace(path):
    ns = {'__name__': '__evaluation__'}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for i, cell in enumerate(json.loads(Path(path).read_text())['cells']):
            if cell['cell_type'] == 'code' and 'experiment' not in cell['metadata'].get('tags', []):
                exec(compile(''.join(cell['source']), f'{path}:cell{i}', 'exec'), ns)
    ns['NOTEBOOK_PATH'] = Path(path).resolve()
    return ns


def frames(ns, datasets):
    result = {}
    if 'main' in datasets:
        result['main'] = ns['load_benchmark']()
    if 'variants' in datasets:
        result['variants'] = ns['load_variants_for_baseline'](ROOT/'benchmark_dataset/questions.csv')
    if 'columns' in datasets:
        result.update(ns['load_redundant_sheets_for_baseline'](
            ROOT/'redundancy_complete/redundant_instances.xlsx', ns['COLUMN_SHEETS']))
    for name, frame in result.items():
        expected = 101 if name == 'main' else 36 if name == 'variants' else 35
        if len(frame) != expected or frame['problem_id'].duplicated().any():
            raise ValueError(f'Invalid case inventory: {name}')
        for address in frame['dataset_address']:
            if any(not Path(p).is_file() for p in address.splitlines()):
                raise FileNotFoundError(address)
    return result


def initialize(path, method):
    global NS
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    NS = namespace(path)
    NS['EVALUATION_METHOD'] = method
    class Recorder(NS['BaseCallbackHandler']):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            if CASE_DIR is not None:
                data = {'utc': datetime.now(timezone.utc).isoformat(),
                        'messages': [[{'role': m.type, 'content': m.content} for m in group] for group in messages]}
                with (CASE_DIR/'prompts.jsonl').open('a') as f:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
        def on_llm_end(self, response, **kwargs):
            if CASE_DIR is not None:
                usage = (response.llm_output or {}).get('token_usage', {})
                with (CASE_DIR/'usage.jsonl').open('a') as f:
                    f.write(json.dumps({'usage': usage}, default=str)+'\n')
    original = NS['make_llm']
    def logged(*args, **kwargs):
        llm = original(*args, **kwargs)
        llm.callbacks = list(llm.callbacks or [])+[Recorder()]
        return llm
    NS['make_llm'] = logged


def worker(item):
    global CASE_DIR
    ns = NS
    folder = Path(item['folder']); CASE_DIR = folder
    started = time.monotonic(); expired = threading.Event()
    (folder/'attempt.json').write_text(json.dumps({'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()}))
    def interrupt():
        expired.set(); os.kill(os.getpid(), signal.SIGINT)
    timer = threading.Timer(DEADLINE, interrupt); timer.daemon = True; timer.start()
    with (folder/'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            if ns['EVALUATION_METHOD'] == 'examples_and_route':
                manifest = ns['set_loto_fold'](item['case']['true_label'])
                (folder/'fold_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            record = ns['execute_pipeline_case'](item['case'])
        except BaseException as exc:
            traceback.print_exc()
            if expired.is_set():
                old = exc; exc = TimeoutError(f'External evaluation deadline {DEADLINE}s exceeded')
                exc.pipeline_context = getattr(old, 'pipeline_context', {})
            record = ns['error_record'](item['case'], exc)
            if ns['EVALUATION_METHOD'] == 'examples_and_route':
                record.update(ns['loto_record_fields']())
        finally:
            timer.cancel()
        record = ns['finalize_record'](record, rel_tol=1e-4, abs_tol=1e-4)
        record.update(experiment_method=ns['EVALUATION_METHOD'], benchmark=item['benchmark'],
                      seconds=round(time.monotonic()-started, 3), evaluated_utc=datetime.now(timezone.utc).isoformat(),
                      notebook_source_sha256=item['notebook_sha'], external_deadline_seconds=DEADLINE,
                      external_deadline_exceeded=expired.is_set())
        (folder/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        (folder/'attempt.json').write_text(json.dumps({'state': 'finished', 'seconds': record['seconds']}))
    return record


def collect(ns, root):
    for p in (root/'attempts').rglob('result.json'):
        ns['save_record'](root/'automatic/results.csv', json.loads(p.read_text()))


def progress(ns, root, inventory):
    records = ns['load_records'](root/'automatic/results.csv')
    rows = []
    for benchmark, frame in inventory.items():
        group = [r for r in records if r.get('benchmark') == benchmark]
        rows.append({'benchmark': benchmark, 'evaluated': len(group), 'expected': len(frame),
                     'solved': sum(r.get('final_ok') is True for r in group),
                     'objective_match': sum(r.get('solution_correct') is True for r in group),
                     'errors': sum(r.get('record_status') == 'error' for r in group)})
    (root/'progress.json').write_text(json.dumps(rows, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['full','rag_only','few_shot_only','examples_and_route'])
    parser.add_argument('--version', required=True)
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--datasets', nargs='+', choices=['main','variants','columns'], default=['main'])
    parser.add_argument('--cohorts', nargs='+', help='Evaluate these complete cohorts from the selected datasets')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args(); os.chdir(ROOT)
    original = Path(args.notebook).resolve(); root = OUT/f'{args.method}_{args.version}'
    root.mkdir(parents=True, exist_ok=True)
    frozen = root/'frozen_notebook.ipynb'
    if frozen.exists() and sha(frozen) != sha(original):
        raise ValueError('Code changed: use a new experiment version')
    if not frozen.exists(): frozen.write_bytes(original.read_bytes())
    ns = namespace(frozen); inventory = frames(ns, args.datasets)
    if args.cohorts:
        if len(args.cohorts) != len(set(args.cohorts)) or set(args.cohorts).difference(inventory):
            raise ValueError('Cohorts must be unique and exist in the selected datasets')
        inventory = {name: frame for name, frame in inventory.items() if name in args.cohorts}
    ns['require_api_key']()
    if args.method in ['rag_only','few_shot_only']:
        inventory = {k: ns['attach_cached_classifications'](v) for k,v in inventory.items()}
    if args.method == 'examples_and_route':
        for label in ns['CLASS_LABELS']:
            manifest = ns['set_loto_fold'](label)
            if manifest['disabled_workflow_route'] in manifest['allowed_routes']:
                raise AssertionError('Forbidden route allowed')
    hashes = {}
    for path in [ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv',
                 ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101.csv',
                 ROOT/'benchmark_dataset/questions.csv', ROOT/'redundancy_complete/redundant_instances.xlsx']:
        hashes[str(path)] = sha(path)
    for frame in inventory.values():
        for address in frame['dataset_address']:
            for path in address.splitlines(): hashes[path] = sha(path)
    if args.method in ['rag_only','few_shot_only']:
        path = ns['CLASSIFICATION_RESULTS_PATH']; hashes[str(path)] = sha(path)
    manifest = {'method': args.method, 'version': args.version, 'source_notebook': str(original),
                'notebook_sha256': sha(frozen), 'input_sha256': hashes,
                'model': ns['MODEL_SNAPSHOT'], 'embedding_model': ns['EMBEDDING_MODEL'],
                'csvqa_modes': ns['CSVQA_MODE_BY_ROUTE'], 'rel_tol': 1e-4, 'abs_tol': 1e-4,
                'external_deadline_seconds': DEADLINE, 'sdk_max_retries': ns['make_llm']().max_retries,
                'selection': 'all declared cases; one pipeline attempt per code version; failures retained',
                'expected': {k: len(v) for k,v in inventory.items()}, 'python': sys.version}
    if args.cohorts:
        manifest['selected_complete_cohorts'] = list(inventory)
    mp = root/'run_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Frozen experiment configuration or input changed')
    if not mp.exists(): mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    collect(ns, root); records = progress(ns, root, inventory)
    if not args.collect_only and any(credit_exhausted(r) for r in records):
        raise ValueError('Previous API quota failures are preserved: use a new experiment version/output round')
    existing = {r['problem_id'] for r in records}; items = []
    for benchmark, frame in inventory.items():
        for case in frame.to_dict('records'):
            if case['problem_id'] in existing: continue
            folder = root/'attempts'/case['problem_id']; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'attempt.json').exists():
                raise ValueError(f'Unfinished previous attempt requires explicit accounting: {folder}')
            items.append({'case': case, 'benchmark': benchmark, 'folder': str(folder), 'notebook_sha': sha(frozen)})
    if args.collect_only: return
    print(f'{args.method} {args.version}: {len(items)} new cases; workers={args.workers}', flush=True)
    quota_exhausted = False
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(frozen),args.method)) as pool:
        futures = {pool.submit(worker,item): item for item in items}
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result(); ns['save_record'](root/'automatic/results.csv', record)
            progress(ns, root, inventory)
            print(f"[{i}/{len(items)}] {record['benchmark']} {record['problem_id']} "
                  f"route={record.get('assigned_route')} solved={record.get('final_ok')} "
                  f"objective_match={record.get('solution_correct')} seconds={record['seconds']} "
                  f"error={record.get('execution_error_type','')}", flush=True)
            if credit_exhausted(record):
                quota_exhausted = True
                for pending in futures:
                    pending.cancel()
                print('API credits exhausted: canceling queued work; retaining results from in-flight attempts.', flush=True)
                break
    if quota_exhausted:
        collect(ns, root)
    records = progress(ns, root, inventory)
    if quota_exhausted:
        (root/'run_status.json').write_text(json.dumps({
            'state': 'externally_blocked', 'reason': 'API credit_balance_exhausted',
            'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'actual_results': len(records),
            'quota_errors': sum(credit_exhausted(r) for r in records),
            'valid_performance_comparison': False,
            'policy': 'Queued work canceled; real in-flight results retained; no synthetic failures. Restore credits and use a new output round.',
        }, ensure_ascii=False, indent=2)+'\n')
        print('Run stopped by API credits; no valid complete performance result.', flush=True)
        raise SystemExit(3)
    with contextlib.redirect_stdout(io.StringIO()): ns['report_results'](records, root/'automatic')
    print('All declared cases recorded.', flush=True)


if __name__ == '__main__':
    main()
