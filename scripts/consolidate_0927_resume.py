"""Combine predeclared whole cohorts of an identical frozen source after API restoration.

No API calls. Original attempt directories remain intact and are linked, not rewritten.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

from evaluate_0927_optimization import OUT as RUNS, namespace, frames, progress

COMPLETED = ['main', 'variants', '50pct-S1', '50pct-S2', '50pct-S3']
REMAINING = [f'{pct}-S{seed}' for pct in ['100pct', '200pct'] for seed in [1, 2, 3]]
CHECK_FIELDS = ['notebook_sha256', 'model', 'embedding_model', 'csvqa_modes',
                'rel_tol', 'abs_tol', 'external_deadline_seconds', 'sdk_max_retries']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consolidate(original, resumed, destination):
    original, resumed, destination = [RUNS/name for name in [original, resumed, destination]]
    assert original != resumed and destination not in {original, resumed}
    manifests = [json.loads((p/'run_manifest.json').read_text()) for p in [original, resumed]]
    for field in CHECK_FIELDS:
        assert manifests[0][field] == manifests[1][field], f'Changed configuration: {field}'
    assert all(m['method'] == 'full' for m in manifests)
    for p,m in zip([original,resumed], manifests):
        assert sha(p/'frozen_notebook.ipynb') == m['notebook_sha256']
        assert all(sha(path) == digest for path,digest in m['input_sha256'].items())
    assert set(manifests[1]['expected']) == set(REMAINING)
    assert all(manifests[1]['input_sha256'][p] == d for p,d in manifests[0]['input_sha256'].items()
               if p in manifests[1]['input_sha256'])
    ns = namespace(original/'frozen_notebook.ipynb')
    inventory = frames(ns, ['main', 'variants', 'columns'])
    selected = {}
    provenance = []
    for folder, cohorts in [(original, COMPLETED), (resumed, REMAINING)]:
        raw = [(p,json.loads(p.read_text())) for p in (folder/'attempts').rglob('result.json')]
        for cohort in cohorts:
            cohort_records = [(p,r) for p,r in raw if r['benchmark'] == cohort]
            expected = inventory[cohort].set_index('problem_id').to_dict('index')
            assert len(cohort_records) == len(expected)
            assert {r['problem_id'] for _,r in cohort_records} == set(expected)
            for path,record in cohort_records:
                assert 'credit_balance_exhausted' not in str(record.get('execution_error','')), 'API quota invalidates a resumed cohort'
                assert record['notebook_source_sha256'] == manifests[0]['notebook_sha256']
                case = expected[record['problem_id']]
                assert record['query'] == case['Query']
                assert record['dataset_address'] == case['dataset_address']
                assert record['label_objective'] == case['label_objective']
                assert ns['finalize_record'](record)['solution_correct'] == record['solution_correct']
                assert record['problem_id'] not in selected
                selected[record['problem_id']] = path
            provenance.append({'benchmark':cohort,'source_run':str(folder),
                'cases':len(cohort_records),'selection':'Entire cohort selected before resumed outcomes; all non-quota failures retained'})
    assert len(selected) == 452
    manifest = {**manifests[0], 'version':destination.name.removeprefix('full_'),
        'source_notebook':str(original/'frozen_notebook.ipynb'),
        'selection':'Five complete original cohorts plus all six resumed cohorts; identical frozen inference source; no per-case selection',
        'cohort_provenance':provenance}
    destination.mkdir(parents=True,exist_ok=True)
    frozen = destination/'frozen_notebook.ipynb'
    if frozen.exists(): assert sha(frozen) == manifests[0]['notebook_sha256']
    else: frozen.write_bytes((original/'frozen_notebook.ipynb').read_bytes())
    control = destination/'run_manifest.json'
    if control.exists(): assert json.loads(control.read_text()) == manifest
    else: control.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for case_id,path in selected.items():
        target = destination/'attempts'/case_id
        target.mkdir(parents=True,exist_ok=True)
        result_path = target/'result.json'
        if result_path.exists(): assert result_path.read_bytes() == path.read_bytes()
        else: result_path.write_bytes(path.read_bytes())
        for artifact in path.parent.iterdir():
            if artifact.name == 'result.json': continue
            link = target/artifact.name
            if link.is_symlink(): assert link.resolve() == artifact.resolve()
            else:
                assert not link.exists(), 'Destination artifacts must link original records'
                link.symlink_to(artifact.resolve(),target_is_directory=artifact.is_dir())
        origin = {'original_result':str(path.resolve()), 'original_sha256':sha(path),
                  'selection':'Whole cohort from predeclared source run'}
        origin_path = target/'record_origin.json'
        if origin_path.exists(): assert json.loads(origin_path.read_text()) == origin
        else: origin_path.write_text(json.dumps(origin,ensure_ascii=False,indent=2)+'\n')
    for path in selected.values():
        ns['save_record'](destination/'automatic/results.csv',json.loads(path.read_text()))
    records = progress(ns,destination,inventory)
    with contextlib.redirect_stdout(io.StringIO()):
        ns['report_results'](records,destination/'automatic')
    (destination/'run_status.json').write_text(json.dumps({'state':'complete',
        'actual_results':len(records),'quota_errors':0,'interrupted_attempts':[],
        'record_access':'Exact original result copies and linked artifacts; record_origin.json and run_manifest.json record provenance'},indent=2)+'\n')
    print(json.dumps({'destination':str(destination),'records':len(records),
        'objective_match':sum(r['solution_correct'] is True for r in records),
        'solved':sum(r['final_ok'] is True for r in records)},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--original',default='full_review_v6')
    parser.add_argument('--resumed',default='full_review_v6_remaining')
    parser.add_argument('--destination',default='full_review_v6_completed')
    args=parser.parse_args()
    consolidate(args.original,args.resumed,args.destination)
