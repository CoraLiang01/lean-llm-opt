"""Read-only audit of preserved GPT-4.1 V1 solutions; never run generated code."""
from pathlib import Path
import ast
import collections
import csv
import hashlib
import json
import math

ROOT = Path('/Users/cora/Documents/GitHub/lean-llm-opt')
BASE = ROOT / 'outputs/leave_one_type_out/gpt41'
OUT = ROOT / 'outputs/loto_cross_model_audit_20261004/gpt41_audit'
LABELS = ['TP', 'NRM', 'RA', 'FLP', 'AP', 'Mixture', 'Others']
ROUTES = ['TP', 'NRM', 'RA', 'FLP', 'AP', 'Others']
SOURCE = BASE / 'V1_review_summary/V1_analysis.json'
previous = json.loads(SOURCE.read_text())
report = {'evidence_limit': 'Current original CSV results are access-protected. Classification and reference-objective values are from the previously verified V1_analysis.json. All 202 current solution.json files and 14 fold manifests were freshly read and validated. Generated programs were inspected statically and never executed.',
          'previous_summary_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'variants': {}}
rows_by_variant = {}

def write_csv(name, rows):
    if not rows:
        return
    # CSV files are currently restricted by the host. Store fresh audit tables
    # as ordinary JSON; never attempt to open or decrypt protected CSV inputs.
    (OUT / name.replace('.csv', '.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2))

def error_kind(error):
    if 'NoneType' in error and 'context manager' in error:
        return 'model_handle_none'
    if 'ModelOutputTruncated' in error:
        return 'output_truncated'
    if 'ragged row' in error:
        return 'legacy_csv_ragged_row'
    if "KeyError: 'model'" in error:
        return 'model_handle_missing'
    if 'keyformat' in error:
        return 'unsupported_gurobi_keyword'
    if 'solver status: 3' in error:
        return 'infeasible_generated_model'
    if 'context_length_exceeded' in error:
        return 'context_length_exceeded'
    if 'APITimeoutError' in error:
        return 'api_timeout'
    if 'Could not match customer' in error:
        return 'customer_column_mismatch'
    if 'could not convert string to float' in error:
        return 'numeric_parse_failure'
    return '' if not error else 'other'

for variant in ['examples_only', 'examples_and_route']:
    folder = BASE / (variant + '_V1')
    prior = previous[variant]['case_details']
    old_rows = {x[0]: dict(zip(prior['columns'], x)) for x in prior['data']}
    manifests = [json.loads((folder / label / 'fold_manifest.json').read_text()) for label in LABELS]
    ids_from_manifests = [pid for manifest in manifests for pid in manifest['case_ids']]
    assert len(ids_from_manifests) == len(set(ids_from_manifests)) == 101
    assert set(ids_from_manifests) == set(old_rows)
    rows = []
    handle_evidence = []
    for manifest in manifests:
        label = manifest['held_out_type']
        for pid in manifest['case_ids']:
            matches = list((folder / label).glob('results_cases/AUTO/' + pid + '/*/solution.json'))
            assert len(matches) == 1, (variant, pid, matches)
            path = matches[0]
            solution = json.loads(path.read_text())
            old = old_rows[pid]
            solved = solution['status'] == 'OPTIMAL'
            objective = solution.get('objective')
            matched = solved and objective is not None and math.isclose(objective, old['label_objective'], rel_tol=1e-4, abs_tol=1e-4)
            error = solution.get('error') or ''
            assert solved == old['final_ok'], (variant, pid, 'solved disagrees')
            assert matched == old['solution_correct'], (variant, pid, 'match disagrees')
            assert error == old['execution_error'], (variant, pid, 'error disagrees')
            assert (objective is None and old['final_objective'] is None) or math.isclose(objective, old['final_objective'], rel_tol=1e-12, abs_tol=1e-8)
            assert label == old['true_label']
            predicted, route = old['predicted_label'], old['assigned_route']
            assert route == ('Others' if predicted == 'Mixture' else predicted)
            assert predicted in manifest['allowed_labels'] and route in manifest['allowed_routes']
            row = {'problem_id': pid, 'true_label': label, 'predicted_label': predicted, 'assigned_route': route,
                   'solved': solved, 'objective_match': matched, 'objective': objective, 'reference_objective': old['label_objective'],
                   'pipeline_stage': old['pipeline_stage'], 'failure_kind': error_kind(error), 'error': error,
                   'solution_path': str(path)}
            rows.append(row)
            if row['failure_kind'] in {'model_handle_none', 'model_handle_missing'}:
                codepath = path.with_name('solve.py')
                tree = ast.parse(codepath.read_text())
                funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
                global_assignments = []
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        target_names = [x.id for x in node.targets if isinstance(x, ast.Name)]
                        if any(x in ('m', 'model') for x in target_names):
                            global_assignments.append(ast.unparse(node))
                handle_evidence.append({'problem_id': pid, 'true_label': label, 'failure_kind': row['failure_kind'],
                  'solve_path': str(codepath), 'global_model_assignments': global_assignments,
                  'function_analysis': [{'name': name,
                    'return_statements': [ast.unparse(n) for n in ast.walk(func) if isinstance(n, ast.Return)],
                    'optimize_calls': [ast.unparse(n) for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'optimize'],
                    'start_line': func.lineno} for name, func in funcs.items()],
                  'interpretation': 'Static code and error confirm no usable global model handle. The stored error has no solver objective; optimality cannot be inferred or credited without a verified rerun.'})
    rows.sort(key=lambda x: x['problem_id'])
    rows_by_variant[variant] = rows
    by_type = []
    classification_matrix, route_matrix = [], []
    for label in LABELS:
        subset = [x for x in rows if x['true_label'] == label]
        n, solved, matched = len(subset), sum(x['solved'] for x in subset), sum(x['objective_match'] for x in subset)
        by_type.append({'true_label': label, 'cases': n, 'solved': solved, 'objective_match': matched, 'failures': n-solved,
                        'solved_mismatch': solved-matched, 'solve_rate': solved/n, 'objective_accuracy': matched/n})
        classification_matrix.append({'true_label': label, **dict.fromkeys(LABELS, 0)})
        route_matrix.append({'true_label': label, **dict.fromkeys(ROUTES, 0)})
        for row in subset:
            classification_matrix[-1][row['predicted_label']] += 1
            route_matrix[-1][row['assigned_route']] += 1
    solved, matched = sum(x['solved'] for x in rows), sum(x['objective_match'] for x in rows)
    summary = {'cases': len(rows), 'unique_ids': len(set(x['problem_id'] for x in rows)), 'solved': solved, 'objective_match': matched,
               'failures': len(rows)-solved, 'solved_mismatch': solved-matched, 'objective_accuracy_among_solved': matched/solved,
               'classification_correct': sum(x['true_label'] == x['predicted_label'] for x in rows) if variant == 'examples_only' else None,
               'forbidden_route_or_label_selections': 0, 'solution_disagreements_with_prior_summary': 0,
               'failure_kinds': dict(collections.Counter(x['failure_kind'] for x in rows if not x['solved']))}
    report['variants'][variant] = {'summary': summary, 'by_type': by_type, 'classification_matrix': classification_matrix,
       'route_matrix': route_matrix, 'manifests': manifests, 'model_handle_static_evidence': handle_evidence}
    for suffix, data in [('cases', rows), ('by_type', by_type), ('classification_matrix', classification_matrix), ('route_matrix', route_matrix)]:
        write_csv(variant + '_' + suffix + '.csv', data)

left = {x['problem_id']: x for x in rows_by_variant['examples_only']}
right = {x['problem_id']: x for x in rows_by_variant['examples_and_route']}
paired = []
for pid in sorted(left):
    a, b = left[pid], right[pid]
    group = 'both_match' if a['objective_match'] and b['objective_match'] else 'gained_match' if b['objective_match'] else 'lost_match' if a['objective_match'] else 'neither_match'
    paired.append({'problem_id': pid, 'true_label': a['true_label'], 'transition': group,
                   'e1_route': a['assigned_route'], 'e2_route': b['assigned_route'],
                   'e1_solved': a['solved'], 'e2_solved': b['solved'], 'e1_match': a['objective_match'], 'e2_match': b['objective_match'],
                   'e1_failure_kind': a['failure_kind'], 'e2_failure_kind': b['failure_kind'],
                   'e1_objective': a['objective'], 'e2_objective': b['objective'], 'reference_objective': a['reference_objective']})
report['paired'] = {'match_transitions': dict(collections.Counter(x['transition'] for x in paired)),
                    'gained_match_origin': dict(collections.Counter(x['e1_failure_kind'] or 'solved_objective_mismatch' for x in paired if x['transition']=='gained_match')),
                    'lost_match_destination': dict(collections.Counter(x['e2_failure_kind'] or 'solved_objective_mismatch' for x in paired if x['transition']=='lost_match')),
                    'solve_transitions': dict(collections.Counter(f"{x['e1_solved']}->{x['e2_solved']}" for x in paired)),
                    'gained': [x for x in paired if x['transition']=='gained_match'],
                    'lost': [x for x in paired if x['transition']=='lost_match']}
common_solved = [x for x in paired if x['e1_solved'] and x['e2_solved']]
report['paired']['common_solved_subset'] = {'cases': len(common_solved),
    'examples_only_objective_matches': sum(x['e1_match'] for x in common_solved),
    'examples_and_route_objective_matches': sum(x['e2_match'] for x in common_solved),
    'interpretation': 'Descriptive comparison on the same successful-execution subset; post-treatment selection means it is not an unbiased causal modeling-effect estimate.'}
write_csv('paired_case_transitions.csv', paired)
(OUT / 'gpt41_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({**{k:v['summary'] for k,v in report['variants'].items()}, 'paired':report['paired']}, ensure_ascii=False, indent=2))
