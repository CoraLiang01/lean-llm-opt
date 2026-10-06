"""Summarize all prespecified paired attempts, including every failure."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/extraction_v6'
manifest = json.loads((OUT / 'sample_manifest.json').read_text())
rows = [json.loads(p.read_text()) for p in (OUT / 'runs').glob('*/*/*/*/result.json')]
assert len(rows) == 36, f'Incomplete experiment: {len(rows)}/36'
index = {(r['version'], r['repeat'], r['dataset'], r['problem_id']): r for r in rows}
assert len(index) == 36


def source_cells(row):
    folder = OUT / 'runs' / row['version'] / f"round_{row['repeat']}" / row['dataset'] / row['problem_id'].replace('/', '_')
    path = folder / 'data_overview.json'
    if not path.exists(): return None
    payload = json.loads(path.read_text())
    cells = {}
    for table in payload['tables']:
        for record in table['records']:
            for column, value in record['values'].items():
                key = (table['file_index'], record['source_row'], column)
                if key in cells: assert cells[key] == value
                cells[key] = value
    return hashlib.sha256(json.dumps(sorted(cells.items()), ensure_ascii=False).encode()).hexdigest()


def paired_outcome(first, second):
    if not first.get('final_ok') or not second.get('final_ok'):
        return first.get('final_ok') == second.get('final_ok') and first.get('execution_error_type') == second.get('execution_error_type')
    a, b = first['final_objective'], second['final_objective']
    return abs(a-b) <= max(1e-4, 1e-4*max(abs(a), abs(b)))


summary = {'runs': len(rows), 'versions': {}, 'cases': [], 'paired_improvements': [], 'paired_regressions': []}
for version in ['baseline', 'improved']:
    subset = [r for r in rows if r['version'] == version]
    pairs = [[index[(version, repeat, sample['dataset'], sample['case']['problem_id'])] for repeat in [1, 2]] for sample in manifest['samples']]
    summary['versions'][version] = {
        'runs': len(subset), 'matched': sum(r.get('solution_correct') is True for r in subset),
        'solved': sum(r.get('final_ok') is True for r in subset),
        'both_rounds_matched_cases': sum(all(r.get('solution_correct') is True for r in pair) for pair in pairs),
        'same_objective_or_error_cases': sum(paired_outcome(*pair) for pair in pairs),
        'same_selected_source_cells_cases': sum(source_cells(pair[0]) is not None and source_cells(pair[0]) == source_cells(pair[1]) for pair in pairs),
        'planner_statuses': dict(Counter(r['csvqa_status'] for r in subset)),
        'repair_attempted': sum(r['repair_attempted'] for r in subset),
        'truncation_retries': sum(r['truncation_retry_count'] for r in subset),
        'median_seconds': round(statistics.median(r['seconds'] for r in subset), 2),
        'sum_tokens': sum(r['llm_usage']['total_tokens'] for r in subset),
        'groups': {group: {'matched': sum(r.get('solution_correct') is True for r in subset if r['dataset'] == group),
                          'runs': sum(r['dataset'] == group for r in subset)} for group in ['101', 'variants', 'columns']},
        'routes': {route: {'matched': sum(r.get('solution_correct') is True for r in subset if r['route'] == route),
                          'runs': sum(r['route'] == route for r in subset)} for route in ['NRM', 'RA']}}

for sample in manifest['samples']:
    entry = {'dataset': sample['dataset'], 'problem_id': sample['case']['problem_id'], 'route': sample['route']}
    for version in ['baseline', 'improved']:
        pair = [index[(version, repeat, sample['dataset'], entry['problem_id'])] for repeat in [1, 2]]
        entry[version] = {'matched': sum(r.get('solution_correct') is True for r in pair),
                          'objectives': [r.get('final_objective') for r in pair],
                          'statuses': [r['csvqa_status'] for r in pair],
                          'rows': [r['extracted_rows'] for r in pair],
                          'errors': [r.get('execution_error') for r in pair]}
    summary['cases'].append(entry)
    for repeat in [1, 2]:
        base = index[('baseline', repeat, sample['dataset'], entry['problem_id'])]
        improved = index[('improved', repeat, sample['dataset'], entry['problem_id'])]
        if not base.get('solution_correct') and improved.get('solution_correct'):
            summary['paired_improvements'].append({'repeat': repeat, **{k: entry[k] for k in ['dataset', 'problem_id']}})
        if base.get('solution_correct') and not improved.get('solution_correct'):
            summary['paired_regressions'].append({'repeat': repeat, **{k: entry[k] for k in ['dataset', 'problem_id']}})
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

base, improved = summary['versions']['baseline'], summary['versions']['improved']
text = f'''# GPT-4.1 结构化提取副本：修改与抽样实验

比较对象是当前主版本 v5 与本次副本 V6，主 notebook 未修改。不是与旧版 V2 的比较。
9 个预先选定的输入，两版各独立运行两轮，共 36 次完整的“取数—建模—代码生成—求解”。
全部已记录，包括失败；没有按目标值修复代码、选择性重跑或复用上一轮模型结果。

## 结果

| 指标 | 当前主版本 v5 | 副本 V6 |
|---|---:|---:|
| 目标匹配 | {base['matched']}/18 | {improved['matched']}/18 |
| 获得最优解 | {base['solved']}/18 | {improved['solved']}/18 |
| 两轮均匹配的输入 | {base['both_rounds_matched_cases']}/9 | {improved['both_rounds_matched_cases']}/9 |
| 两轮目标或错误类型一致 | {base['same_objective_or_error_cases']}/9 | {improved['same_objective_or_error_cases']}/9 |
| 两轮选出的源单元格一致 | {base['same_selected_source_cells_cases']}/9 | {improved['same_selected_source_cells_cases']}/9 |
| 完整数据回退 | {base['planner_statuses'].get('FALLBACK_FULL_DATA',0)}/18 | {improved['planner_statuses'].get('FALLBACK_FULL_DATA',0)}/18 |
| 单次中位耗时 | {base['median_seconds']} 秒 | {improved['median_seconds']} 秒 |
| 记录的对话 token 合计 | {base['sum_tokens']} | {improved['sum_tokens']} |

“目标或错误类型一致”包含稳定地算错，不能当成正确率。源单元格一致性忽略角色标签、列序和视图分组，按物理文件、源行号、列名和值比较。耗时包含网络和检索，有并发影响；token 为客户端回调记录，不包括 embedding，不能等同账单金额。

| 数据组 | v5 匹配 | V6 匹配 | 测试范围 |
|---|---:|---:|---|
'''
for group, scope in [('101', '2 个 NRM、2 个 RA，各两轮'), ('variants', '2 个扩展题强制 RA，各两轮'), ('columns', '最后三个 sheet 各 1 个 RA，各两轮')]:
    a, b = base['groups'][group], improved['groups'][group]
    text += f"| {group} | {a['matched']}/{a['runs']} | {b['matched']}/{b['runs']} | {scope} |\n"
text += '\n| 输入 | 路线 | v5 匹配轮数 | V6 匹配轮数 | v5 两轮目标 | V6 两轮目标 |\n|---|---|---:|---:|---|---|\n'
for entry in summary['cases']:
    text += f"| {entry['dataset']} / {entry['problem_id']} | {entry['route']} | {entry['baseline']['matched']}/2 | {entry['improved']['matched']}/2 | {entry['baseline']['objectives']} | {entry['improved']['objectives']} |\n"
text += f'''
逐轮配对中，副本由失败变匹配：{len(summary['paired_improvements'])} 次；由匹配变失败：{len(summary['paired_regressions'])} 次。
NRM：v5 {base['routes']['NRM']['matched']}/4，V6 {improved['routes']['NRM']['matched']}/4。
RA：v5 {base['routes']['RA']['matched']}/14，V6 {improved['routes']['RA']['matched']}/14。

## 关键个案与结论

改善集中在 `200pct-S2/OR-009`：v5 两轮都只抽取 Queens、Brooklyn 两项，将原题的举例误读成筛选范围；副本两轮均抽取全部 20 项。
v5 第一轮使用连续变量，目标 288.0859538784067；第二轮使用整数变量，但仍只有两项，目标 0。
副本两轮使用整数变量，目标均为参考值 3912。副本改了 schema、证据校验、参数映射和提示，当前实验不能把收益单独归因于某一项。
证据检查也不能自动证明“举例”与“限定子集”的语义：即使 JSON 与 evidence 合法，模型仍可能选错范围。

NRM 两版均 4/4 匹配，未显示正确率提升。副本 `OR-013` 第一轮出现一次 `finish_reason=length`，单次建模输出达到 32768 token；经已有 NRM 重试后成功，耗时 188.31 秒。第二轮未截断，耗时 12.23 秒。
因此不能说副本让 NRM 更稳定或更省成本。两版取数的源单元格重复一致性都为 9/9，本次也没有证明抽取一致性进一步提高。

Variant29 的副本两轮均因将不兼容关系 `item_a -> item_b` 绑定成一对一映射而被拒绝：`item_a` 可出现在多条不兼容关系中，需要复合键，不能作为唯一键。
原始取数保留关系行，不代表会发生字典覆盖；新增 bindings 的校验暴露的是模型声明的映射不成立。默认关闭计划 repair，因此回退到全部数据继续建模；两轮都匹配，但没有实现预期的结构化计划成功率改善。
两版这里都算对，不能把回退本身计为精度收益。

整体上副本在这组样本的 RA 目标匹配和重复正确结果上更好；收益只有一题，两轮重复不能算作两个独立问题的证据。
对话 token 合计增加约 {round((improved['sum_tokens']/base['sum_tokens']-1)*100,1)}%，其中包含上述截断，以及更详细的 schema/bindings 提示和复杂题完整数据回退。
建议保留副本继续验证，而不是直接把本次结果推广为全量、所有路线或总体稳定性提升。

## 实际修改

1. 提取计划使用 `with_structured_output(CSVExtractionPlan, method="json_schema", strict=True, include_raw=True)`。
   Pydantic 规定固定字段、枚举、禁止额外字段；所有字段必填，不需要日期格式时传 null。
   移除“从自由文本中找首尾大括号”的解析。结构化输出不可解析时按原有失败流程处理，记录原始输出。
2. 添加 `bindings`：明确源参数对应的 `table_id`、`value_column`、`index_columns`。
   Python 检查表和列存在、复合键不为空且不重复、无索引的标量视图只有一行。
   账本/分量行使用完整业务键保留原始分量；所需求和和单位换算仍由建模及代码阶段完成。
   没有执行 GPT 生成的取数代码，也没有在提取阶段计算、改写系数。
3. 过滤器增加类型和取值形状校验：数值不能用 prefix/contains，eq 等不能误传一串行数相同的列表，between 必须两个边界。
   evidence 必须在原题中出现，允许空白与大小写差异，但不再靠删除标点或碰巧出现的数值绕过证据检查。
4. 矩阵轴使用完整原始 ID 匹配。`A-1` 与 `A1`、`001` 与 `1`、不同大小写的 ID 不再自动视为相同。
   不改变源行顺序，不排序重排来偷偷对齐。
5. NRM/RA 建模提示及 planned 代码提示读取 bindings，保留复合键，区分源字段与派生总量。
   原题决定数学含义；bindings 校验通过不代表参数语义一定正确。
6. 副本使用独立版本号、结果目录，默认不自动运行整套实验；保留主版本及其结果。

CSV 仍由 `_load_tables()` 读取完整源表，再按计划确定性抽取。本次没有加入 `usecols` 分块读取或任意 Python/SQL 取数程序，未验证 I/O 性能提升。

示例：
```json
{{
  "parameter": "profit",
  "table_id": "file_1_view_0",
  "value_column": "Value",
  "index_columns": ["ProductName"]
}}
```
多选背包的索引则为 `["Family", "Option"]`，不会将各家族的 `O1` 合并。

流程：原题与源表 profile → 严格计划 → Python 校验与完整抽取 → CSVQA_DATA（含 bindings）→ 原有 ReAct 抽象模型 → 原有 RAG 代码生成 → Gurobi 求解与接口验证。

## 开关和实验控制

- 两版 `CSVQA_REPAIR_PLAN_ON_FAILURE=False`：本次没有让 GPT 修提取计划。失败直接完整数据回退。
- 副本保留该开关：打开后仍是“校验失败 → GPT 按同一严格 schema 修一次 → 再校验 → 失败则完整数据回退”。
- 两版 `NORMALIZE_GUROBI_BULK_NAMES=False`，保留生成的变量和约束名称。
- 两版保留 API `max_retries=2`、NRM 截断后一次建模重试、既有 ReAct 协议纠错；没有新增求解报错后的代码 repair。
- 同一模型快照、原题、CSV、RAG 语料；只共享检索索引，不共享模型输出。逐题交替版本先后顺序，第一轮完成后再运行第二轮。
- Gurobi 两线程、180 秒上限，两版一致；没有将非最优 incumbent 当作最优。
- 标准匹配容差：`rel_tol=1e-4, abs_tol=1e-4`；参考答案只用于本地评分。

## 能说明什么

样本按结构覆盖预先选择，不是随机代表性抽样；每题两轮不足以证明总体稳定性或全量提升。
variants 和冗余列没有 NRM 标签。variants 的 Variant6 为 Others、Variant29 为 Mixture；强制 RA 测的是扩展建模能力，不能与原来的自动分类全量成绩直接比较。
本次没有修改分类器、RAG 示例内容、其它路线或参考答案。严格 JSON 和源键校验能避免一类接口错误，但不能保证变量域、目标或约束的语义正确。

## 可复核文件

- `sample_manifest.json`：开始前固定的样本、模型、开关和输入/代码哈希。
- `baseline_v5.ipynb`、`improved_v6.ipynb`：实验冻结的两个版本。
- `results.csv`、`results.json`、`summary.json`：全部尝试和统计。
- `runs/<版本>/round_<轮次>/<数据组>/<题号>/`：计划、抽取数据、抽象模型、生成代码、求解结果和日志。
- `verification.json`、`tests_red.log`、`tests_green.log`：41 项本地检查与修改前失败证据。
- `changes.diff`：代码差异。
'''
(OUT / '实验与修改说明.md').write_text(text)
print(json.dumps(summary['versions'], ensure_ascii=False, indent=2))
