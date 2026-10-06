"""Frozen paired NRM/RA sample, two fresh rounds. Gold answers are local scoring only."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/extraction_v6'
BASE = OUT / 'baseline_v5.ipynb'
COPY = ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_Extraction_V6.ipynb'
IMPROVED = OUT / 'improved_v6.ipynb'
NS = {}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def namespace(book):
    os.environ['LEAN_LLM_OPT_ROOT'] = str(ROOT)
    ns = {'__name__': 'sample_evaluation'}
    for i, cell in enumerate(json.loads(book.read_text())['cells']):
        if cell['cell_type'] == 'code' and i < 37:
            exec(compile(''.join(cell['source']), f'{book}:cell{i}', 'exec'), ns)
    ns['NOTEBOOK_PATH'] = book
    return ns


def select_cases(ns):
    frames = {
        '101': ns['load_benchmark'](ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101.csv'),
        'variants': ns['load_benchmark'](ROOT / 'benchmark_dataset 3/questions.csv',
                                        dataset_root=ROOT / 'benchmark_dataset 3'),
        'columns': ns['load_benchmark'](ROOT / 'columns.xls')}
    frames['variants']['problem_id'] = frames['variants']['dataset_address'].map(
        lambda addr: next(p for p in Path(addr.splitlines()[0]).parts if p.startswith('Variant')))
    selection = [
        ('101', 'OR-010', 'NRM', 'all products, source IDs'),
        ('101', 'OR-013', 'NRM', '47932-row source, explicit category subset'),
        ('101', 'OR-035', 'RA', 'one capacity, integer quantities'),
        ('101', 'OR-039', 'RA', 'independent warehouse capacities'),
        ('variants', 'Variant6', 'RA', 'binary multi-choice, compound family-option keys, two resources'),
        ('variants', 'Variant29', 'RA', 'multiple resources, unit conversions, ledgers, fixed fees and relationships'),
        ('columns', '200pct-S1/OR-005', 'RA', 'renamed numeric columns, administrative distractors'),
        ('columns', '200pct-S2/OR-009', 'RA', 'ambiguous wording with redundant numeric columns'),
        ('columns', '200pct-S3/OR-012', 'RA', 'renamed IDs and multiple independent resource capacities')]
    keep = {'problem_id', 'Query', 'dataset_address', 'true_label', 'true_route',
            'label_objective', 'benchmark_sheet'}
    cases = []
    for group, ident, route, reason in selection:
        found = frames[group].loc[frames[group]['problem_id'].eq(ident)]
        assert len(found) == 1, (group, ident)
        case = {key: value for key, value in found.iloc[0].to_dict().items() if key in keep}
        assert case['label_objective'] is not None
        cases.append({'dataset': group, 'route': route, 'reason': reason, 'case': case})
    return cases


def initialize():
    global NS
    NS = {name: namespace(book) for name, book in [('baseline', BASE), ('improved', IMPROVED)]}
    # Identical RAG corpus/retrieval algorithm; share only the index, never model responses.
    NS['improved']['_get_rag_store'] = NS['baseline']['_get_rag_store']
    NS['baseline']['gp'].setParam('Threads', 2)
    NS['baseline']['gp'].setParam('TimeLimit', 180)


def work(item):
    from langchain_community.callbacks.manager import get_openai_callback
    repeat, number, sample = item
    versions = ['baseline', 'improved'] if (number + repeat) % 2 else ['improved', 'baseline']
    rows = []
    for version in versions:
        case, ns = sample['case'], NS[version]
        folder = OUT / 'runs' / version / f'round_{repeat}' / sample['dataset'] / case['problem_id'].replace('/', '_')
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'result.json').exists():
            rows.append(json.loads((folder / 'result.json').read_text())); continue
        observed_states = []
        original = ns['build_csvqa_components']
        def capture(*args, **kwargs):
            components = original(*args, **kwargs)
            observed_states.append(components[2])
            return components
        ns['build_csvqa_components'] = capture
        start = time.monotonic()
        with (folder / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), get_openai_callback() as usage:
            try:
                record = ns['execute_pipeline_case'](case, forced_route=sample['route'])
            except Exception as exc:
                traceback.print_exc()
                record = ns['error_record'](case, exc, sample['route'])
            record = ns['finalize_record'](record)
        ns['build_csvqa_components'] = original
        traces = [state['trace'] for state in observed_states if state.get('trace')]
        observation = record.get('csvqa_observation') or next(
            (state.get('observation') for state in reversed(observed_states) if state.get('observation')), '')
        if observation: record['csvqa_observation'] = observation
        if traces:
            record['csvqa_trace'] = json.dumps(traces[-1], ensure_ascii=False, indent=2)
        for field, name in [('generated_model', 'model.md'), ('solve_code', 'solve.py'),
                            ('csvqa_observation', 'data_overview.json'), ('csvqa_trace', 'csvqa_trace.json')]:
            if record.get(field) is not None: (folder / name).write_text(str(record[field]))
        (folder / 'all_csvqa_traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2))
        if record.get('final_solution') is not None:
            (folder / 'solution.json').write_text(json.dumps(record['final_solution'], ensure_ascii=False))
        data = json.loads(observation) if observation else {}
        # Data hash excludes semantic labels/bindings; compares actual selected source data.
        selected_data = [{'file_index': t['file_index'], 'columns': t['columns'],
                          'records': t['records']} for t in data.get('tables', [])]
        row = {key: value for key, value in record.items() if key not in
               {'generated_model', 'solve_code', 'csvqa_observation', 'csvqa_trace', 'final_solution'}}
        row.update(version=version, repeat=repeat, dataset=sample['dataset'], route=sample['route'],
                   seconds=round(time.monotonic() - start, 2),
                   csvqa_status=traces[-1]['status'] if traces else None,
                   planner_attempts=sum(t.get('planner_attempt_count', 0) for t in traces),
                   repair_attempted=any(t.get('planner_repair_attempted') for t in traces),
                   truncation_retry_count=max(0, len(observed_states) - 1),
                   extracted_rows=[t['returned_rows'] for t in data.get('tables', [])],
                   extracted_columns=[t['columns'] for t in data.get('tables', [])],
                   selected_data_hash=hashlib.sha256(json.dumps(selected_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest() if selected_data else None,
                   planner_errors=[error for t in traces for error in t.get('planner_errors', [])],
                   binding_count=len(data.get('bindings', [])),
                   llm_usage={'requests': usage.successful_requests, 'prompt_tokens': usage.prompt_tokens,
                              'completion_tokens': usage.completion_tokens, 'total_tokens': usage.total_tokens},
                   source_sha256=sha(BASE if version == 'baseline' else IMPROVED))
        (folder / 'result.json').write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str, allow_nan=False))
        print(json.dumps({'done': version, 'round': repeat, 'dataset': row['dataset'],
                          'id': row['problem_id'], 'route': row['route'],
                          'matched': row.get('solution_correct'), 'plan': row['csvqa_status'],
                          'seconds': row['seconds']}, ensure_ascii=False), flush=True)
        rows.append(row)
    return rows


def collect():
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in (OUT / 'runs').glob('*/*/*/*/result.json')]
    (OUT / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if rows: pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False, encoding='utf-8-sig')
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    assert (ROOT / 'LEAN_LLM_OPT_4.1_Large-scale.ipynb').read_bytes() == BASE.read_bytes()
    if not IMPROVED.exists(): IMPROVED.write_bytes(COPY.read_bytes())
    assert IMPROVED.read_bytes() == COPY.read_bytes()
    ns = namespace(BASE); samples = select_cases(ns)
    paths = {ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101.csv', ROOT / 'columns.xls',
             ROOT / 'benchmark_dataset 3/questions.csv'}
    paths.update((ROOT / 'Large_Scale_Or_Files').rglob('*.csv'))
    for sample in samples:
        paths.update(Path(raw) for raw in sample['case']['dataset_address'].splitlines())
    manifest = {'sample_policy': 'prespecified structural coverage; not a random representative sample',
                'models': {'chat': ns['MODEL_SNAPSHOT'], 'embedding': ns['EMBEDDING_MODEL']},
                'routing': 'forced NRM/RA; variants labels are Others/Mixture',
                'fresh_repeats': 2, 'pipeline_runs': 36, 'code_repair': False, 'plan_repair': False,
                'normalize_bulk_names': False, 'nrm_truncation_retry': True,
                'gurobi_threads': 2, 'gurobi_time_limit_seconds': 180,
                'shared_retrieval_index': True, 'gold_sent_to_model': False,
                'baseline_sha256': sha(BASE), 'improved_sha256': sha(IMPROVED),
                'inputs': {str(p): sha(p) for p in sorted(paths)}, 'samples': samples}
    path = OUT / 'sample_manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == manifest, 'Frozen experiment changed'
    else: path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'preflight': 'PASS', 'samples': len(samples), 'runs': 36, 'api_run': args.run}), flush=True)
    if not args.run: return
    if not args.resume: assert not list((OUT / 'runs').glob('*/*/*/*/result.json')), 'Use --resume; failures are never rerun'
    items = [(repeat, i, sample) for repeat in [1, 2] for i, sample in enumerate(samples)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        # Complete round one before dispatching round two; do not change the code between rounds.
        for repeat in [1, 2]:
            futures = [pool.submit(work, item) for item in items if item[0] == repeat]
            for future in as_completed(futures):
                future.result(); rows = collect()
                print(json.dumps({'recorded_runs': len(rows), 'total': 36}), flush=True)
    rows = collect(); assert len(rows) == 36
    unchanged = all(sha(Path(p)) == digest for p, digest in manifest['inputs'].items())
    source_unchanged = sha(BASE) == manifest['baseline_sha256'] and sha(IMPROVED) == manifest['improved_sha256']
    (OUT / 'completion.json').write_text(json.dumps({'runs': len(rows), 'inputs_unchanged': unchanged,
        'frozen_sources_unchanged': source_unchanged, 'main_unchanged': sha(ROOT / 'LEAN_LLM_OPT_4.1_Large-scale.ipynb') == sha(BASE)}, indent=2))
    assert unchanged and source_unchanged


if __name__ == '__main__': main()
