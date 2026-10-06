"""Pair three V2 sample runs with their unchanged historical outcomes by case ID.

An absent V2 result is reported as pending, never counted as a failed solve.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpt41_model_interface_v2_validation'
AUDIT = ROOT / 'outputs/loto_cross_model_audit_20261004/gpt41_audit'
CASE_IDS = ['OR-002', 'OR-004', 'OR-018', 'OR-036', 'OR-063', 'OR-069']
VARIANTS = ('full', 'examples_only', 'examples_and_route')


def historical_rows():
    example_cases = {row['problem_id']: row for row in json.loads((AUDIT / 'examples_only_cases.json').read_text())}
    old = {}
    for variant in VARIANTS:
        if variant == 'full':
            for case_id in CASE_IDS:
                files = list((ROOT / 'outputs/final_101_V2/automatic/results_cases/AUTO' / case_id).glob('*/solution.json'))
                if len(files) != 1:
                    raise ValueError(f'Expected one original full-run solution for {case_id}; found {len(files)}')
                saved = json.loads(files[0].read_text())
                reference = float(example_cases[case_id]['reference_objective'])
                solved = saved.get('status') == 'OPTIMAL'
                objective = saved.get('objective') if solved else None
                old[variant, case_id] = dict(true_label=example_cases[case_id]['true_label'],
                    solved=solved, objective=objective,
                    objective_match=bool(solved and math.isclose(float(objective), reference, rel_tol=1e-4, abs_tol=1e-4)),
                    reference_objective=reference,
                    assigned_route=None, error=saved.get('error'))
        else:
            cases = {row['problem_id']: row for row in json.loads((AUDIT / f'{variant}_cases.json').read_text())}
            for case_id in CASE_IDS:
                row = cases[case_id]
                if row['true_label'] != example_cases[case_id]['true_label']:
                    raise ValueError(f'Historical case label differs across variants: {case_id}')
                old[variant, case_id] = dict(true_label=row['true_label'], solved=bool(row['solved']),
                    objective=row['objective'], objective_match=bool(row['objective_match']),
                    reference_objective=float(row['reference_objective']),
                    assigned_route=row['assigned_route'], error=row['error'])
    return old


def saved_sample_rows(directory, variant):
    root = Path(directory)
    files = [root / 'results.csv'] if variant == 'full' else list(root.glob('*/results.csv'))
    found = {}
    for path in files:
        if not path.is_file():
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                case_id = row['problem_id']
                if case_id not in CASE_IDS:
                    raise ValueError(f'Unexpected sample case in {path}: {case_id}')
                if case_id in found:
                    raise ValueError(f'Duplicate sample case in {directory}: {case_id}')
                found[case_id] = row
    return found


def compare(status=None):
    if status is None:
        status = json.loads((OUT / 'end_to_end_sample_status.json').read_text())
    old = historical_rows()
    pairs, summaries = [], []
    for variant in VARIANTS:
        item = next((entry for entry in status if entry['notebook'].startswith(
            'LEAN_LLM_OPT_4.1' if variant == 'full' else
            'LOTO_Examples_Only' if variant == 'examples_only' else 'LOTO_Examples_And_Route')), None)
        if item is None:
            raise ValueError(f'Missing sample status for {variant}')
        new = saved_sample_rows(item['output_dir'], variant) if item.get('output_dir') else {}
        expected_labels = {case_id: old[variant, case_id]['true_label'] for case_id in CASE_IDS}
        for case_id, row in new.items():
            if row.get('true_label') != expected_labels[case_id]:
                raise ValueError(f'New true label does not match archived label: {variant}/{case_id}')
            original = old[variant, case_id]
            if row.get('label_objective') and not math.isclose(float(row['label_objective']), original['reference_objective'], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'Reference objective changed: {variant}/{case_id}')
        for case_id in CASE_IDS:
            original = old[variant, case_id]
            row = new.get(case_id)
            solved = row.get('final_ok') == 'True' if row else None
            objective = float(row['final_objective']) if row and row.get('final_objective') else None
            matched = bool(solved and objective is not None and math.isclose(
                objective, original['reference_objective'], rel_tol=1e-4, abs_tol=1e-4)) if row else None
            if row and row.get('solution_correct') in ('True', 'False') and matched != (row['solution_correct'] == 'True'):
                raise ValueError(f'Stored match flag differs from recomputed comparison: {variant}/{case_id}')
            pairs.append(dict(variant=variant, problem_id=case_id, true_label=original['true_label'],
                reference_objective=original['reference_objective'],
                historical_route=original['assigned_route'], historical_solved=original['solved'],
                historical_objective=original['objective'], historical_match=original['objective_match'],
                new_status='recorded' if row else 'pending',
                new_predicted_label=row.get('predicted_label') if row else None,
                new_selected_route=row.get('selected_route') if row else None,
                new_executed_route=row.get('executed_route') if row else None,
                new_solved=solved, new_objective=objective, new_match=matched,
                new_model_contract_recovered=row.get('model_contract_recovered') if row else None,
                new_failure_category=row.get('failure_category') if row else None,
                new_error=row.get('execution_error') if row else None))
        group = [pair for pair in pairs if pair['variant'] == variant]
        summaries.append(dict(variant=variant, expected_cases=len(CASE_IDS), new_recorded=len(new),
            historical_solved=sum(pair['historical_solved'] for pair in group),
            historical_match=sum(pair['historical_match'] for pair in group),
            new_solved=sum(pair['new_solved'] for pair in group) if len(new) == len(CASE_IDS) else None,
            new_match=sum(pair['new_match'] for pair in group) if len(new) == len(CASE_IDS) else None,
            completeness='complete' if len(new) == len(CASE_IDS) else 'pending'))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'end_to_end_comparison.json').write_text(json.dumps(
        {'summary': summaries, 'pairs': pairs, 'note': 'Pending V2 cases are excluded from new-score totals.'},
        ensure_ascii=False, indent=2))
    lines = ['# GPT-4.1 三组 V2 抽样与历史结果配对', '',
             '同一批六道题。历史全量来源为 `outputs/final_101_V2/automatic`；两个 LOTO 历史来源分别为 `examples_only_V1` 和 `examples_and_route_V1`。新结果按问题 ID 配对，目标值用相同的 `1e-4` 容差重算。', '',
             '| 条件 | 历史求解 | 历史匹配 | V2 已记录 | V2 求解 | V2 匹配 | 状态 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f'| {s["variant"]} | {s["historical_solved"]}/6 | {s["historical_match"]}/6 | '
                     f'{s["new_recorded"]}/6 | {str(s["new_solved"]) if s["new_solved"] is not None else "—"} | '
                     f'{str(s["new_match"]) if s["new_match"] is not None else "—"} | {s["completeness"]} |')
    lines += ['', '完整的新成绩只在同一条件六道题全部保存后显示；尚未运行的题不会记为失败。', '',
              '| 条件 | 题号 | 历史 route | 历史求解 | 历史匹配 | V2 route | V2 求解 | V2 匹配 | 模型对象恢复 |',
              '|---|---|---|---|---|---|---|---|---|']
    for row in pairs:
        def status_value(value): return '待运行' if value is None else '是' if value is True else '否' if value is False else str(value)
        lines.append('| ' + ' | '.join(status_value(row[key]) for key in
            ('variant','problem_id','historical_route','historical_solved','historical_match',
             'new_selected_route','new_solved','new_match','new_model_contract_recovered')) + ' |')
    (OUT / 'END_TO_END_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return summaries


if __name__ == '__main__':
    compare()
