"""Report every version/case, including unsuccessful attempts and paired changes."""
import csv
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_0927_optimization import namespace

OUT = ROOT/'outputs/optimization_0927_20261006'
REPORT = OUT/'report'


def write_csv(path, records, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


def main():
    REPORT.mkdir(exist_ok=True)
    summaries = []; details = []; paired = []; inventories = {}; runtime_stats = []
    historical = namespace(OUT/'source_snapshots/before/full_before.ipynb')
    old_main = historical['load_records'](ROOT/'outputs/final_101_V2/automatic/results.csv')
    old_by_id = {r['problem_id']: r for r in old_main}
    for root in sorted(OUT.iterdir()):
        if not (root/'run_manifest.json').exists(): continue
        manifest = json.loads((root/'run_manifest.json').read_text())
        ns = namespace(root/'frozen_notebook.ipynb')
        records = ns['load_records'](root/'automatic/results.csv')
        inventories[root.name] = records
        attempts = [json.loads(path.read_text()) for path in (root/'attempts').rglob('result.json')]
        runtime_stats.append({'run': root.name, 'completed_attempts': len(attempts),
            'sdk_retries': sum(int(r.get('api_retry_count') or 0) for r in attempts),
            'external_case_timeouts': sum(r.get('external_deadline_exceeded') is True for r in attempts),
            'error_types': dict(Counter(r['execution_error_type'] for r in attempts if r.get('execution_error_type')))})
        declared = manifest.get('expected', {'main': manifest.get('expected_cases', 101)})
        version = manifest.get('version', root.name.removeprefix(manifest['method']+'_'))
        for benchmark, expected in declared.items():
            group = [r for r in records if r.get('benchmark', 'main') == benchmark]
            solved = sum(r.get('final_ok') is True for r in group)
            correct = sum(r.get('solution_correct') is True for r in group)
            target = (32 if benchmark == 'variants' else 31 if benchmark != 'main' else None) if manifest['method'] == 'full' else None
            summaries.append({'run': root.name, 'method': manifest['method'], 'version': version,
                'benchmark': benchmark, 'evaluated': len(group), 'total': expected,
                'solved': solved, 'objective_match': correct,
                'solved_rate': solved/expected, 'objective_accuracy': correct/expected,
                'target': target, 'target_met': correct >= target if len(group) == expected and target is not None else None,
                'errors': sum(r.get('record_status') == 'error' for r in group),
                'mismatches': sum(r.get('final_ok') is True and r.get('solution_correct') is not True for r in group),
                'status': 'complete' if len(group) == expected else 'in progress'})
        for record in records:
            row = {'run': root.name, **{k: record.get(k) for k in ['benchmark','problem_id','assigned_route',
                'true_label','predicted_label','disabled_workflow_route','final_ok','solution_correct',
                'final_objective','label_objective','pipeline_stage','execution_error_type','execution_error',
                'seconds','api_retry_count','external_deadline_exceeded']}}
            row['attempt_directory'] = str(root/'attempts'/record['problem_id'])
            row['benchmark'] = record.get('benchmark', 'main')
            row['outcome'] = 'correct' if record.get('solution_correct') is True else 'objective mismatch' if record.get('final_ok') is True else 'failed'
            details.append(row)
    before_by_id = {r['problem_id']: r for r in inventories.get('full_before', [])}
    before_by_id.update(old_by_id)
    for run, records in inventories.items():
        if not run.startswith('full_') or run == 'full_before': continue
        for record in records:
            previous = before_by_id.get(record['problem_id'])
            if previous is None: continue
            paired.append({'run': run, 'benchmark': record['benchmark'], 'problem_id': record['problem_id'],
                'before_solved': previous.get('final_ok'), 'after_solved': record.get('final_ok'),
                'before_correct': previous.get('solution_correct'), 'after_correct': record.get('solution_correct'),
                'before_objective': previous.get('final_objective'), 'after_objective': record.get('final_objective')})
    write_csv(REPORT/'summary.csv', summaries, ['run','method','version','benchmark','evaluated','total','solved',
        'objective_match','solved_rate','objective_accuracy','target','target_met','errors','mismatches','status'])
    detail_fields = ['run','benchmark','problem_id','outcome','assigned_route','true_label','predicted_label',
        'disabled_workflow_route','final_ok','solution_correct','final_objective','label_objective','pipeline_stage',
        'execution_error_type','execution_error','seconds','api_retry_count','external_deadline_exceeded','attempt_directory']
    write_csv(REPORT/'all_cases.csv', details, detail_fields)
    write_csv(REPORT/'unsuccessful_cases.csv', [r for r in details if r['outcome'] != 'correct'], detail_fields)
    write_csv(REPORT/'paired_changes.csv', paired, ['run','benchmark','problem_id','before_solved','after_solved',
        'before_correct','after_correct','before_objective','after_objective'])
    old_methods = {}
    for method in ['rag_only', 'few_shot_only', 'examples_and_route']:
        previous = historical['load_records'](ROOT/f'outputs/ablation_loto_0927/{method}_v1/automatic/results.csv')
        old_methods[method] = {'total': len(previous),
            'solved': sum(r.get('final_ok') is True for r in previous),
            'objective_match': sum(r.get('solution_correct') is True for r in previous)}
    full_main = next((r for r in summaries if r['run'] == 'full_v1' and r['benchmark'] == 'main'), None)
    chosen_runs = {'rag_only': 'rag_only_final', 'few_shot_only': 'few_shot_only_final_v2',
                   'examples_and_route': 'examples_and_route_final'}
    method_comparisons = []
    for method, run in chosen_runs.items():
        current = next((r for r in summaries if r['run'] == run and r['benchmark'] == 'main'), None)
        if current is None: continue
        old = old_methods[method]
        complete = current['status'] == 'complete'
        method_comparisons.append({'method': method, 'run': run, 'status': current['status'],
            'evaluated': current['evaluated'], 'total': current['total'],
            'objective_match': current['objective_match'], 'solved': current['solved'],
            'before_objective_match': old['objective_match'], 'before_solved': old['solved'],
            'objective_delta_before': current['objective_match']-old['objective_match'] if complete else None,
            'solved_delta_before': current['solved']-old['solved'] if complete else None,
            'objective_delta_full': current['objective_match']-full_main['objective_match'] if complete and full_main else None,
            'solved_delta_full': current['solved']-full_main['solved'] if complete and full_main else None,
            'objective_pp_delta_full': 100*(current['objective_match']-full_main['objective_match'])/101 if complete and full_main else None})
    write_csv(REPORT/'method_comparisons.csv', method_comparisons, ['method','run','status','evaluated','total',
        'objective_match','solved','before_objective_match','before_solved','objective_delta_before',
        'solved_delta_before','objective_delta_full','solved_delta_full','objective_pp_delta_full'])
    (REPORT/'runtime_stats.json').write_text(json.dumps(runtime_stats, ensure_ascii=False, indent=2))
    example_frame = historical['read_csv_compat'](ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv',
        dtype=str, keep_default_na=False)
    removed_examples = [{'row_index': int(i), 'semantic_type': row['Type'],
        'workflow_route': historical['class_to_workflow_route'](historical['normalize_problem_class'](row['Type'])),
        'prompt': row['prompt'], 'reference_data_address': row['Data_address']}
        for i,row in example_frame.iterrows()]
    write_csv(REPORT/'rag_removed_examples.csv', removed_examples,
        ['row_index','semantic_type','workflow_route','prompt','reference_data_address'])
    gates = {}
    for run in inventories:
        groups = [r for r in summaries if r['run'] == run]
        if run.startswith('full_'):
            gates[run] = {'all_declared_cases_completed': all(r['status'] == 'complete' for r in groups),
                'all_additional_targets_met': all(r['target_met'] is True for r in groups if r['benchmark'] != 'main'),
                'has_new_101_evaluation': any(r['benchmark'] == 'main' and r['evaluated'] == 101 for r in groups)}
    (REPORT/'target_gates.json').write_text(json.dumps(gates, indent=2))
    lines = ['# 0927 最小修改与完整评测记录', '',
        '正确统一使用 Objective match：Gurobi 最优目标值与参考答案按 rel_tol=abs_tol=1e-4 比较。Solved 单独统计。失败、超时和不匹配全部保留，分母使用完整题数。', '',
        '修改前主数据集基准来自已经验证对应代码/数据的 final_101_V2：Objective match 95/101，Solved 99/101；修改前 Variants 和九个冗余列 sheet 在本轮重新执行。', '',
        '| 运行版本 | 数据集 | 完成 | Objective match | Solved | 目标 |',
        '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['run']} | {row['benchmark']} | {row['evaluated']}/{row['total']} | {row['objective_match']}/{row['total']} | {row['solved']}/{row['total']} | {row['target'] or '—'} |")
    if any(r['status'] != 'complete' for r in summaries):
        lines += ['', '**仍有实验未完成；未完成行是进度，不能作为最终准确率。**']
    lines += ['', '## 最终方法与全量版本比较', '',
        '两份消融复用 full_v1 的 101 个分类标签和 route（分类正确 94/101）。LOTO 每题在排除目标语义类别样本、禁用对应 route 后重新分类。', '',
        'Few-shot Only 最终采用 final_v2：完整数据在建模阶段提供，代码生成函数入口只接受结构信息。这项边界修正是在运行 final_v2 之前确定的，与准确率无关。中间版 few_shot_only_final 的完整结果单独保留，未逐题选择较好结果。', '',
        '| 方法 | 本轮 Objective match | 本轮 Solved | 修改前 Objective match / Solved | 相对本轮全量正确题数变化 |',
        '|---|---:|---:|---:|---:|']
    for row in method_comparisons:
        delta = f"{row['objective_delta_full']:+d}" if row['objective_delta_full'] is not None else '待完成'
        lines.append(f"| {row['method']} | {row['objective_match']}/{row['total']} | {row['solved']}/{row['total']} | {row['before_objective_match']}/{row['total']} / {row['before_solved']}/{row['total']} | {delta} |")
    lines += ['', '## 主集回归', '',
        '历史主集 Objective match 95/101，本轮 full_v1 为 92/101（−3 题、−2.97 个百分点）；Solved 99/101 → 96/101。原先正确的 OR-027、OR-041、OR-082、OR-085 变错，原先输出截断的 OR-013 变对，其余 96 题正确性不变。这五题的 route 均未改变。', '',
        '- OR-027：数据包含 Organic Fruits / Organic Staples / Organic Vegetables，本轮生成代码却用 organ 精确相等筛选，导致空集。历史版本使用 prefix/contains 筛选。',
        '- OR-041：生成模型从整数变量改为连续变量；3912 → 4408.523076923077，Objective match 失败。题目中的 scale 用语未明确整数域，不能依据参考目标值反向指定变量类型。',
        '- OR-082：生成代码对 pandas Series 调用 casefold，而非通过字符串访问器调用，出现 AttributeError。',
        '- OR-085：itertuples 会转换非法 Python 字段名，生成代码仍用原列名 Unnamed: 0 取属性，出现 AttributeError。',
        '- 多项提示词在同一版本中调整；主集对照是历史运行而非同一时刻重跑原版。以上可以确定直接失败原因，不能据此隔离每项改动的因果效应或把全部变化归因于运行波动。', '']
    lines += ['## 组件边界', '',
        '### RAG Only', '',
        '从所有 route 的建模与代码生成提示中删除以下 15 个训练参考示例（行号从 0 起）。同时停用参考示例的检索入口；分类仍使用最终全量预测缓存。具体题干和数据路径见 rag_removed_examples.csv。', '',
        '| 参考行 | 语义类别 | 建模与代码示例使用的 route |',
        '|---|---|---|']
    for row in removed_examples:
        lines.append(f"| {row['row_index']} | {row['semantic_type']} | {row['workflow_route']} |")
    lines += ['',
        '保留本题数据的 CSVQA Tool、CSV 行文档及其向量检索、legacy 数据检索和抽取、来源字段解析、NRM 原有 planned 数据计划及 Python 执行，以及 Others 的字段信息读取和求解时完整 CSV 读取。删除的是训练参考示例，不是全部 RAG 数据功能。', '',
        '### Few-shot Only', '',
        '保留各 route 的训练示例及其建模/代码参考检索。删除本题 CSVQA Tool、LLM 数据检索与抽取计划；Python 按 dtype=str、keep_default_na=False 读取本题全部 CSV 行、列和单元格，以完整 Observation 交给建模 agent。建模阶段保持 ReAct 解析器和执行器，当前任务工具列表为空。', '',
        '代码生成函数仅接收建模输出、问题、结构信息和保留的训练示例；原始完整 CSV Observation 不进入该函数或其提示。完整数据仅在后续 Python 求解执行时提供。Others 同样在建模时使用完整数据、代码生成时仅使用结构信息。正常建模输出及参考示例中出现的数值不等于重新传入完整原始 CSV payload。', '',
        '本轮还观察到生成波动：中间 Few-shot 版本为 89/101，最终严格边界版本为 86/101；两次运行 100/101 题的首次建模提示完全相同，正确性变化的 7 题首次建模提示也均相同。不能将这 3 题净变化解释为数据边界修正的因果效应，比较记录见 fewshot_boundary_prompt_comparison.json。最终始终使用完整 final_v2 结果。', '',
        '### LOTO Examples And Route', '',
        '按真实语义类别划分 fold，仅用于示例排除与禁用边界。分类 RefData 和建模/代码参考库均先删除目标语义类别，再进行 route 合并；UFLP 归一为 FLP。Mixture 和 Others 共用 Others route，因此二者 fold 都禁用 Others route 及其输出标签。分类 agent 每题在允许标签中重新分类，真实类别不用于指定替代 route。', '',
        '分类、建模、参考检索和代码生成均设禁用 route 检查；参考模型和参考目标值只用于事后评价。最终运行边界审计见 final_boundary_audit.json。', '']
    lines += ['', '## 修改范围', '']
    for scope in sorted(OUT.glob('revision_scope_*.json')):
        data = json.loads(scope.read_text()); lines += [f"### {data['revision']}", '']
        for change in data['changes']: lines.append('- '+change)
        lines += ['', '改动依据案例：'+', '.join(data.get('evidence_cases', []))+'.', '']
    lines += ['实际局部改动说明：', '',
        '- HTTP：共享连接池，将统一 SDK 最大重试从 0 改为 1，180 秒模型请求超时保持原值。修改前扩展集出现连接错误；每个版本均记录实际重试。full_v1 的 452 题实际 SDK 重试为 0，不能将其提高的正确数归因于成功重试。',
        '- legacy 来源：行文档和抽取结果保留 source；抽取提示要求单个 JSON 数组，减少标题、Markdown 和省略号导致的解析问题。',
        '- 约束语义：明确固定启用费用不会自动将无条件数量下限改为条件下限，依据 Variant32 的失败。',
        '- Others：只压缩抽象计划的重复、枚举内容，保持七个章节和所有约束，依据 Variant1 的输出截断。',
        '- 代码：添加变量容器不覆盖、quicksum 参数类型、字符串读取 CSV 后显式转数值的提示，依据 Variant28/35 等生成错误。',
        '- AP：将不可用配对作为决策掩码处理，不能因不可用配对存在成本值而拒绝数据，依据 Variant27。',
        '- 模型接口：要求在模块级暴露存活的 m 模型，不把 optimize() 返回值当模型。执行器本身不变。',
        '- 框架和模式：ReAct 不变，只有原有 NRM planned，其余仍为 legacy；模型快照、采样参数、求解器指令和评分容差未修改。', '']
    lines += ['## 复现与审计', '',
        '- 每个运行目录包含 frozen_notebook.ipynb 和 run_manifest.json，保存代码及输入哈希、模型设置和完整题目清单。',
        '- attempts 中保留每题 run.log、prompts.jsonl、attempt.json 和 result.json；通用版本运行器另存 usage.jsonl，LOTO 另存 fold_manifest.json。',
        '- all_cases.csv 列出所有案例；unsuccessful_cases.csv 列出所有失败、超时及目标值不匹配；paired_changes.csv 对照每题修改前后的结果。',
        '- SDK 对瞬时服务/网络失败的统一重试属于同一次 pipeline，事件另行记录；没有根据目标值更换结果、排除案例或重跑单题。', '']
    audit_path = OUT/'final_boundary_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ['### 本轮边界检查', '',
            '- 交付全量 notebook 的函数 AST（包括字符串常量）及提示词、route 模式常量与 full_v1 冻结评测版本一致；交付文件和输入哈希均已核对。',
            '- 两份消融完整 101 题的分类标签、route、题干、数据路径均与最终全量分类缓存一致。',
            '- Few-shot Only 已核对 101 份完整 Python Observation，共 289332 个本题原始 CSV 单元格；代码生成入口拒绝原始 records。输入过长导致的模型调用失败仍计入结果。',
            '- LOTO 完整 101 题均在允许 route 上执行；七个 fold 的目标示例排除及执行前禁用检查通过。',
            '- 各版本所有案例的失败类型、SDK 重试和外部超时见 runtime_stats.json；RAG Only 有 4 个 API 请求超时案例。', '']
    lines += ['### 交付代码', '']
    names = [
        ('全量', 'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'),
        ('RAG Only', 'Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb'),
        ('Few-shot Only', 'Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb'),
        ('LOTO Examples And Route', 'LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb')]
    for label, name in names: lines.append(f'- [{label}]({ROOT/name})')
    lines += ['', f'- [修改前全量原始归档]({OUT}/source_snapshots/before/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb)',
        f'- [逐题完整结果]({REPORT}/all_cases.csv)',
        f'- [所有失败、超时及目标值不匹配]({REPORT}/unsuccessful_cases.csv)',
        f'- [主集及扩展集逐题前后对照]({REPORT}/paired_changes.csv)',
        f'- [最终组件边界审计]({audit_path})', '']
    (REPORT/'report.md').write_text('\n'.join(lines))
    print(json.dumps(gates, indent=2)); print(REPORT/'report.md')


if __name__ == '__main__': main()
