"""Aggregate saved GPT-4.1 LOTO folds without invoking models or solvers.

Run with the repository's Python environment. Raw experiment files are read-only.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
LABELS = ['TP', 'NRM', 'RA', 'FLP', 'AP', 'Mixture', 'Others']
ROUTES = ['TP', 'NRM', 'RA', 'FLP', 'AP', 'Others']
VARIANTS = ['examples_only', 'examples_and_route']
BOOKS = [f'LOTO_{variant}_{model}_Large-scale.ipynb'
         for model in ['GPT4.1', 'gpt_oss_20b']
         for variant in ['Examples_Only', 'Examples_And_Route']]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    if value in ('True', 'true', True, '1', 1):
        return True
    if value in ('False', 'false', False, '0', 0):
        return False
    if value in ('', None):
        return None
    raise ValueError(f'Unexpected boolean {value!r}')


def number(value):
    if value in ('', None):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def route(label):
    return 'Others' if label in ('Others', 'Mixture') else label


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(escape, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(escape, row)) + ' |' for row in rows])


def rate(n, d):
    return f'{n}/{d}（{n / d:.2%}）' if d else '不适用'


def fmt(value):
    return '—' if value is None else f'{value:.12g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/01a0fb27_loto_summary_20261003')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    details = []
    summaries, per_type, group_rows, route_rows, error_groups = [], [], [], [], []
    for exp, variant in enumerate(VARIANTS, 1):
        root = ROOT / 'outputs/leave_one_type_out/gpt41' / variant
        source_rows = []
        for label in LABELS:
            source = root / label / 'results.csv'
            source_hashes[str(source.relative_to(ROOT))] = sha(source)
            rows = read_csv(source)
            assert all(r['held_out_type'] == r['true_label'] == label for r in rows)
            assert all(r['loto_variant'] == variant for r in rows)
            source_rows.extend(rows)
        source_rows.sort(key=lambda r: r['problem_id'])
        assert [r['problem_id'] for r in source_rows] == [f'OR-{i:03d}' for i in range(1, 102)]
        summary_path = root / 'case_summary.csv'
        source_hashes[str(summary_path.relative_to(ROOT))] = sha(summary_path)
        saved_summary = {r['problem_id']: r for r in read_csv(summary_path)}
        assert len(saved_summary) == 101
        for original in source_rows:
            assert all(original.get(k, '') == v
                       for k, v in saved_summary[original['problem_id']].items()), original['problem_id']
            row = dict(original)
            for key in ['execution_error_type', 'execution_error', 'csvqa_mode', 'csvqa_status']:
                row.setdefault(key, '')
            row['experiment'] = exp
            for key in ['final_ok', 'solution_correct', 'classification_correct', 'route_allowed']:
                row[key] = boolean(row.get(key))
            for key in ['final_objective', 'label_objective']:
                row[key] = number(row.get(key))
            assert row['label_objective'] is not None
            matched = bool(row['final_ok'] and row['final_objective'] is not None and
                           math.isclose(row['final_objective'], row['label_objective'],
                                        rel_tol=1e-4, abs_tol=1e-4))
            assert matched == row['solution_correct']
            assert row['assigned_route'] == route(row['predicted_label'])
            row['allowed_label_verified'] = row['predicted_label'] in row['allowed_labels'].split(',')
            row['allowed_route_verified'] = row['assigned_route'] != row['disabled_workflow_route']
            assert row['allowed_label_verified'] and row['allowed_route_verified']
            row['true_route_match'] = (row['assigned_route'] == route(row['true_label'])) if exp == 1 else None
            assert row['classification_correct'] == (row['true_label'] == row['predicted_label']) if exp == 1 else row['classification_correct'] is None
            row['outcome'] = ('目标匹配' if matched else '已求解但目标不匹配' if row['final_ok'] else '执行失败')
            row['absolute_objective_error'] = (abs(row['final_objective'] - row['label_objective'])
                                               if row['final_objective'] is not None else None)
            row['relative_objective_error'] = (row['absolute_objective_error'] / abs(row['label_objective'])
                                               if row['absolute_objective_error'] is not None and row['label_objective'] != 0 else None)
            row['source_results'] = str(root / row['held_out_type'] / 'results.csv')
            for key in ['solution_path', 'generated_model_path', 'solve_code_path', 'csvqa_observation_path']:
                row[key + '_absolute'] = str(root / row['held_out_type'] / row[key]) if row.get(key) else ''
            details.append(row)
        subset = [r for r in details if r['experiment'] == exp]
        n, solved, matched = len(subset), sum(r['final_ok'] for r in subset), sum(r['solution_correct'] for r in subset)
        for label in LABELS:
            group = [r for r in subset if r['true_label'] == label]
            per_type.append(dict(experiment=exp, true_label=label, n=len(group),
                                 classification_correct=sum(bool(r['classification_correct']) for r in group) if exp == 1 else None,
                                 solved=sum(r['final_ok'] for r in group), matched=sum(r['solution_correct'] for r in group)))
        macro = sum(r['matched'] / r['n'] for r in per_type if r['experiment'] == exp) / len(LABELS)
        summaries.append(dict(experiment=exp, variant=variant, n=n,
                              classification_correct=sum(bool(r['classification_correct']) for r in subset) if exp == 1 else None,
                              true_route_match=sum(bool(r['true_route_match']) for r in subset) if exp == 1 else None,
                              solved=solved, matched=matched, failed=n-solved,
                              solved_not_matched=solved-matched, solve_rate=solved/n,
                              objective_match_rate=matched/n, macro_objective_match_rate=macro,
                              match_rate_among_solved=matched/solved,
                              forbidden_route_hits=sum(not r['allowed_route_verified'] for r in subset)))
        groups = defaultdict(list)
        for r in subset:
            groups[(r['true_label'], r['predicted_label'], r['assigned_route'])].append(r)
        for (label, prediction, assigned), members in sorted(groups.items(), key=lambda p: (LABELS.index(p[0][0]), LABELS.index(p[0][1]))):
            group_rows.append(dict(experiment=exp, true_label=label, predicted_label=prediction,
                                   assigned_route=assigned, n=len(members), solved=sum(r['final_ok'] for r in members),
                                   matched=sum(r['solution_correct'] for r in members),
                                   problem_ids=', '.join(r['problem_id'] for r in members)))
        for assigned in ROUTES:
            members = [r for r in subset if r['assigned_route'] == assigned]
            route_rows.append(dict(experiment=exp, assigned_route=assigned, n=len(members),
                                   solved=sum(r['final_ok'] for r in members), matched=sum(r['solution_correct'] for r in members)))
        groups = defaultdict(list)
        for r in subset:
            if not r['final_ok']:
                groups[(r['pipeline_stage'], r['execution_error_type'])].append(r)
        for (stage, error), members in sorted(groups.items()):
            error_groups.append(dict(experiment=exp, pipeline_stage=stage, execution_error_type=error,
                                     n=len(members), problem_ids=', '.join(r['problem_id'] for r in members)))
    by_exp = {exp: {r['problem_id']: r for r in details if r['experiment'] == exp} for exp in (1, 2)}
    paired = []
    for pid, a in by_exp[1].items():
        b = by_exp[2][pid]
        for key in ['true_label', 'label_objective', 'query', 'dataset_address']:
            assert a[key] == b[key], (pid, key)
        change = ('两组均匹配' if a['solution_correct'] and b['solution_correct'] else
                  '实验2新增匹配' if b['solution_correct'] else
                  '实验2丢失匹配' if a['solution_correct'] else '两组均不匹配')
        row = dict(problem_id=pid, true_label=a['true_label'], label_objective=a['label_objective'], match_change=change)
        for exp, r in [(1, a), (2, b)]:
            for key in ['predicted_label', 'assigned_route', 'final_ok', 'solution_correct', 'final_objective',
                        'outcome', 'pipeline_stage', 'execution_error_type']:
                row[f'exp{exp}_{key}'] = r[key]
        paired.append(row)
    notebooks = {name: json.loads((ROOT/name).read_text()) for name in BOOKS}
    code_hashes = {name: {str(i): hashlib.sha256(''.join(book['cells'][i]['source']).encode()).hexdigest()
                          for i in (37, 41)} for name, book in notebooks.items()}
    for i in ('37', '41'):
        assert len({h[i] for h in code_hashes.values()}) == 1
    provenance = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      source_sha256=source_hashes, notebook_sha256={name: sha(ROOT/name) for name in BOOKS},
                      shared_cell_sha256=code_hashes,
                      objective_match_definition='final_ok and math.isclose(final_objective, label_objective, rel_tol=1e-4, abs_tol=1e-4)',
                      checks={'complete_unique_ids_per_experiment': 101, 'fold_vs_case_summary': 'all cells equal',
                              'paired_input_fields': 'equal', 'objective_matches_recomputed': True,
                              'label_route_mapping': 'all correct', 'disabled_route_hits_exp2': 0})
    payload = dict(provenance=provenance, summaries=summaries, per_type=per_type, details=details,
                   paired=paired, classification_routes=group_rows, routes=route_rows, errors=error_groups)
    (out/'analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, rows in [('summary', summaries), ('per_type', per_type), ('case_details', details),
                       ('paired_comparison', paired), ('classification_routes', group_rows),
                       ('route_summary', route_rows), ('failure_groups', error_groups),
                       ('failed_cases', [r for r in details if not r['final_ok']]),
                       ('objective_mismatches', [r for r in details if r['final_ok'] and not r['solution_correct']])]:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        write_csv(out/f'{name}.csv', rows, fields)
    report = ['# GPT-4.1 Leave-one-type-out 实验结果与 oss20b 协议核对',
              f"统计时间：{provenance['generated_at']}。从两实验各七折的 `results.csv` 重新聚合，并与条件目录的 `case_summary.csv` 逐单元格核对。两组均完整覆盖 101 个唯一题号，没有重复或缺失。本次仅分析保存结果，没有重新调用模型或求解器。",
              '## 总体结果',
              '实验 1：删除当前真实语义类别在 `RAG_Examples_All.csv` 中的示例，保留全部 route。实验 2：同样删除示例，并禁用该类别对应的执行 route。',
              table(['指标', '实验 1', '实验 2'], [
                  ['语义分类正确', rate(summaries[0]['classification_correct'],101), '不适用：真值类别不可输出'],
                  ['与真实类别对应的 route 一致', rate(summaries[0]['true_route_match'],101), '不适用：该 route 被禁用'],
                  ['完成求解', *[rate(s['solved'],s['n']) for s in summaries]],
                  ['目标值匹配（所有题为分母）', *[rate(s['matched'],s['n']) for s in summaries]],
                  ['已求解题中的目标值匹配', *[rate(s['matched'],s['solved']) for s in summaries]],
                  ['目标匹配率：七类宏平均', *[f"{s['macro_objective_match_rate']:.2%}" for s in summaries]],
                  ['执行失败', *[s['failed'] for s in summaries]],
                  ['已求解但目标不匹配', *[s['solved_not_matched'] for s in summaries]]]),
              '“完成求解”采用保存结果的 `final_ok`；它表示流水线取得求解结果，不等于模型一定正确。目标值匹配使用 `math.isclose(rel_tol=1e-4, abs_tol=1e-4)`，失败题计为不匹配；没有额外检查所有约束、变量语义或解的完整正确性。不同问题的目标单位不同，目标误差不跨题直接平均。',
              '## 按真实类别汇总',
              table(['真实类别', '题数', '实验1 分类正确', '实验1 求解成功', '实验1 目标匹配', '实验2 求解成功', '实验2 目标匹配'],
                    [[label, per_type[i]['n'], rate(per_type[i]['classification_correct'],per_type[i]['n']),
                      rate(per_type[i]['solved'],per_type[i]['n']),rate(per_type[i]['matched'],per_type[i]['n']),
                      rate(per_type[i+7]['solved'],per_type[i+7]['n']),rate(per_type[i+7]['matched'],per_type[i+7]['n'])]
                     for i,label in enumerate(LABELS)]),
              '## 分类标签与执行 route',
              '分类输出是七个语义标签；执行层只有六个 route。`Mixture` 和 `Others` 都映射为 `Others`。因此语义标签和执行 route 分开统计。下表的“选择 route”来自 `assigned_route`，包含后续执行失败的题目，并不表示这些题都求解成功。']
    for exp in (1,2):
        report += [f'### 实验 {exp}：真实类别 → 预测标签 → 选择 route',
                   table(['真实类别','预测标签','选择 route','题数','求解成功','目标匹配'],
                         [[r[k] for k in ['true_label','predicted_label','assigned_route','n','solved','matched']]
                          for r in group_rows if r['experiment']==exp])]
    report += ['实验 1 的 8 个语义误分类中，有 2 题是 `Mixture→Others`，仍进入相同 route，所以语义分类正确为 93 题，route 与真值对应一致为 95 题。实验 2 的全部 14 道 FLP 被预测为 `Mixture`、执行 `Others`；22 道 RA 中 21 题预测 `Others`、1 题预测 `Mixture`，全部执行 `Others`。',
               '### 按选择 route 聚合',
               table(['route','实验1 题数/求解/匹配','实验2 题数/求解/匹配'],
                     [[assigned, *[' / '.join(str(r[k]) for k in ['n','solved','matched'])
                                   for r in route_rows if r['assigned_route']==assigned]] for assigned in ROUTES]),
               '两组全部 101 题的预测标签均属于当折允许集合；实验 2 禁用 route 使用次数为 0（包含失败题）。原 `route_allowed` 字段仅成功记录为 True，失败记录为空，不能把空值解读为越过禁用规则。明细另给出独立重算的 `allowed_route_verified`。',
               '## 失败及目标值不匹配',
               table(['实验','失败阶段','错误类型','题数','题号'],
                     [[r[k] for k in ['experiment','pipeline_stage','execution_error_type','n','problem_ids']] for r in error_groups]),
               '实验 1 有 19 个求解阶段 `TypeError`，错误文本均为 `NoneType object does not support the context manager protocol`（原文 NoneType 带引号）。抽查 OR-008、OR-011、OR-035，生成的封装函数没有返回模型，外部 `m = solve_*()` 得到 None，后续模型上下文执行失败；尚未逐一修复全部代码。实验 2 的 9 个代码生成阶段 ValueError 是 RA Observation 行长度不一致，另有 3 个求解 ValueError 和 1 个上下文超限 BadRequestError。',
               '### 已求解但目标值不匹配的题目',
               table(['实验','题号','真实类别','预测标签','选择 route','所得目标值','参考目标值'],
                     [[r['experiment'],r['problem_id'],r['true_label'],r['predicted_label'],r['assigned_route'],fmt(r['final_objective']),fmt(r['label_objective'])]
                      for r in details if r['final_ok'] and not r['solution_correct']]),
               '## 两实验逐题配对',
               table(['配对结果','题数','题号'],
                     [[change,len([r for r in paired if r['match_change']==change]),
                       ', '.join(r['problem_id'] for r in paired if r['match_change']==change)]
                      for change in ['两组均匹配','实验2新增匹配','实验2丢失匹配','两组均不匹配']]),
               '实验 2 净增加 14 道完成求解、8 道目标匹配，但同时有 13 道原本匹配的题失去匹配。21 道新增匹配全部来自实验 1 的执行失败：19 道上述 TypeError，2 道输出截断（OR-020、OR-038）。因此可以报告这次端到端结果提高，不能据此认定禁用专用 route 提高了建模能力；代码执行可靠性是明显影响因素。已求解子集内的目标匹配率反而从 97.30% 降到 90.91%。这是当前保存运行的描述性比较，没有多次独立运行或不确定性估计。',
               '## oss20b 与 GPT-4.1 的实验协议一致性',
               '四份 notebook 的 LOTO 控制单元（零基索引 37）和 runner/汇总单元（零基索引 41）逐字相同，已保存源文件及单元 SHA-256。离线回归另验证真实 101 题及参考 CSV，但使用模拟模型、向量索引和求解器。',
               table(['检查项','对齐规则'],[
                   ['七折与样本','TP 9、NRM 25、RA 22、FLP 14、AP 5、Mixture 18、Others 8；每题一次'],
                   ['示例删除','按原始语义 Type 过滤；每折删除行数 1、1、1、1、1、8、2；Others/Mixture 不互删'],
                   ['过滤范围','分类、建模及代码生成中使用同一 CSV 的入口均读过滤版本；空示例返回空集合'],
                   ['实验1','保留全部 route 和基线分类提示'],
                   ['实验2','禁用同类物理 route；Others/Mixture 两折都禁用 Others route 及两个语义标签'],
                   ['分类限制','给原提示追加动态候选；oss 的 parser 修正和补全提示也使用该候选集合'],
                   ['硬保护','dispatch 前拒绝禁用 route，记录 route_selection 错误；不自动改路或新增 General'],
                   ['缓存','换折清空分类/示例缓存；结果按条件、折、源代码/配置/数据指纹隔离'],
                   ['续跑','只复用已完成求解且指纹/产物校验通过记录；目标错误仍可复用，失败重算'],
                   ['评价','同一目标匹配容限；失败计入分母；实验2分类正确率不适用']]),
               '对齐指实验控制协议一致，各模型保留各自全量基线。GPT-4.1 仅 NRM 使用 planned CSVQA；oss20b 五个具体 route 使用 planned。模型、embedding、分类解析/补全、上下文预算、代码规范化、求解设置及原有产物缓存校验仍有基线差异，不宣称两模型端到端实现或调用预算完全相同。GPT-4.1 产物复用校验包含内容哈希，oss 延用原有可读性校验。',
               '当前 GPT-4.1 两份的 RUN_LOTO=True，oss20b 两份为 False；这是运行开关，不是实验设计差异。本次未启动 oss20b 推理，当前结果目录也没有 oss20b 成绩。审计未发现需要修改 notebook 实验协议的差异。',
               '### 对论文表述的边界',
               '两实验删除的是指定 CSV 的同类示例，库外手写示例、七类定义和跨类别结构相近示例仍保留。真实标签不作为分类器的逐题参数，但用它分折并构造实验2的禁用集合，因此实验2的提示间接提供了本折排除信息。实验2是预先禁用真值 route 的受限路由压力测试，不应称为完全无类别先验的未见类型泛化。候选标签通过提示约束，非模型层强制解码；硬保证来自 dispatch 前的拒绝检查。',
               '## 文件与复核',
               '- `GPT41_LOTO_results.xlsx`：总体/类别/route汇总、202 条逐题记录和 101 条配对记录，可筛选。',
               '- `case_details.csv`：全部原始结果字段及重算标记，含题干、错误文本、模型/代码/解文件绝对路径。',
               '- `classification_routes.csv`：各类别→预测标签→route 的题数、求解/匹配数和全部题号。',
               '- `paired_comparison.csv`、`failed_cases.csv`、`objective_mismatches.csv`：逐题变化及异常明细。',
               '- `provenance.json`：输入文件哈希、共同单元哈希、时间及核对口径；`analysis.json` 为结构化汇总。',
               '原始实验输出及四份 notebook 本次均保持不变。复核读取所有折，避免只使用部分续跑产生的 summary。此前 APIConnectionError 重跑后的当前记录已包含在内：OR-012 和 OR-066 匹配，OR-013 当前为建模输出截断；当前两组没有 APIConnectionError。']
    (out/'GPT41_LOTO_report.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(out), 'summaries':summaries,
                      'paired_counts':dict(Counter(r['match_change'] for r in paired)),
                      'detail_rows':len(details)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
