"""Revalidate saved extraction plans only; no generation, optimization or scoring."""
import argparse
import contextlib
import io
import json

import evaluate_react_revision_20261006 as runner


def main(source_version, validator_version):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ns = runner.namespace(runner.ROOT / runner.NAMES['full'], 'full')
    groups = {'automatic': ns['load_benchmark'](), 'variants': ns['load_variants_for_baseline'](
        runner.ROOT / 'benchmark_dataset/questions.csv')}
    groups.update({'columns/' + key: value for key, value in ns['load_redundant_sheets_for_baseline'](
        runner.ROOT / 'redundancy_complete/redundant_instances.xlsx',
        [f'{p}-S{s}' for p in ('50pct', '100pct', '200pct') for s in (1, 2, 3)]).items()})
    # Copy only source/query fields. Reference models, labels and objectives are not inspected.
    inputs = {(group, row['problem_id']): {'query': row['Query'], 'address': row['dataset_address']}
              for group, frame in groups.items() for row in frame.to_dict('records')}
    folder = runner.OUT / f'full_{source_version}'
    checked, rejected = 0, []
    for path in folder.glob('**/attempts/*/result.json'):
        record = json.loads(path.read_text())
        trace = json.loads(record.get('csvqa_trace') or '{}')
        if trace.get('status') != 'PLANNED' or not trace.get('plan'):
            continue
        group = str(path.relative_to(folder)).split('/attempts/')[0]
        current = inputs[group, record['problem_id']]
        tables = ns['_load_tables'](current['address'], file_indices=trace.get('requested_file_indices'))
        checked += 1
        try:
            ns['_execute_plan'](trace['plan'], tables, current['query'], record['assigned_route'])
        except Exception as exc:
            rejected.append({'dataset': group, 'problem_id': record['problem_id'], 'error': str(exc)})
    artifact = {'source_version': source_version, 'validator_version': validator_version,
                'validator_notebook_sha256': runner.sha(runner.ROOT / runner.NAMES['full']),
                'saved_planned_observations_checked': checked, 'newly_rejected_count': len(rejected),
                'rejected': rejected, 'api_calls': 0, 'solves': 0, 'results_rescored': False,
                'reference_models_or_answers_inspected': False}
    target = runner.OUT / f'saved_plan_fragment_validation_{source_version}_with_{validator_version}.json'
    target.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-version', required=True)
    parser.add_argument('--validator-version', required=True)
    args = parser.parse_args()
    main(args.source_version, args.validator_version)
