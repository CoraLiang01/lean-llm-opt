"""Report complete and partial rounds honestly; compare against the current full_v1 baseline."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/review_0927_20261006'
RUNS = ROOT / 'outputs/optimization_0927_20261006'
REPORT = OUT / 'report'
OLD_METHODS = {'rag_only':'rag_only_final', 'few_shot_only':'few_shot_only_final_v2',
               'examples_and_route':'examples_and_route_final'}


def write_csv(name, rows, fields):
    with (REPORT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def records(folder):
    return [{**json.loads(p.read_text()), '_result_path': str(p)}
            for p in (RUNS/folder/'attempts').rglob('result.json')]


def quota_error(record):
    return 'credit_balance_exhausted' in str(record.get('execution_error', ''))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    control_path = OUT/'staged_delivery_control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    chosen = control['version'] if control else None
    pause_path = OUT/'paused_control.json'
    pause = json.loads(pause_path.read_text()) if pause_path.exists() else None
    folders = ['full_v1'] + sorted(p.name for p in RUNS.glob('full_review_v*') if (p/'run_manifest.json').exists())
    folders += list(OLD_METHODS.values())
    if chosen:
        folders += [f'{method}_{chosen}' for method in OLD_METHODS if (RUNS/f'{method}_{chosen}/run_manifest.json').exists()]
        folders += [name for name in control.get('method_runs', {}).values() if (RUNS/name/'run_manifest.json').exists()]
    folders = list(dict.fromkeys(folders))
    summaries, all_cases, pairs, runtime = [], [], [], []
    inventory, statuses = {}, {}
    baseline = {r['problem_id']:r for r in records('full_v1')}
    for name in folders:
        folder = RUNS/name
        if not (folder/'run_manifest.json').exists(): continue
        manifest = json.loads((folder/'run_manifest.json').read_text())
        status_path = folder/'run_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        statuses[name] = status
        rows = records(name); inventory[name] = rows
        expected = manifest.get('expected', {'main':manifest.get('expected_cases',101)})
        for cohort,total in expected.items():
            group = [r for r in rows if r.get('benchmark','main') == cohort]
            solved = sum(r.get('final_ok') is True for r in group)
            objective = sum(r.get('solution_correct') is True for r in group)
            summaries.append({'run':name, 'method':manifest['method'], 'benchmark':cohort,
                'evaluated':len(group), 'total':total, 'objective_match':objective, 'solved':solved,
                'errors':sum(r.get('final_ok') is not True for r in group),
                'mismatches':solved-objective, 'complete':len(group)==total,
                'quota_errors':sum(quota_error(r) for r in group),
                'run_state':status.get('state', 'complete' if len(rows)==sum(expected.values()) else 'incomplete')})
        runtime.append({'run':name, 'evaluated':len(rows),
            'run_status':status,
            'quota_errors':sum(quota_error(r) for r in rows),
            'sdk_retries':sum(int(r.get('api_retry_count') or 0) for r in rows),
            'external_deadlines':sum(r.get('external_deadline_exceeded') is True for r in rows),
            'legacy_source_fallbacks':sum(r.get('csvqa_status') == 'LEGACY_SOURCE_VALIDATION_FALLBACK' for r in rows),
            'legacy_validated_source_rows':sum(r.get('csvqa_status') == 'LEGACY_VALIDATED_SOURCE_ROWS' for r in rows),
            'error_types':dict(Counter(r.get('execution_error_type','unknown') for r in rows if r.get('final_ok') is not True))})
        for r in rows:
            detail = {'run':name, 'benchmark':r.get('benchmark','main'),
                **{k:r.get(k) for k in ['problem_id','assigned_route','predicted_label','true_label',
                    'disabled_workflow_route','held_out_type','final_objective','label_objective',
                    'pipeline_stage','execution_error_type','execution_error','seconds','api_retry_count',
                    'external_deadline_exceeded','notebook_source_sha256']},
                'objective_match':r.get('solution_correct') is True, 'solved':r.get('final_ok') is True,
                'outcome':'correct' if r.get('solution_correct') is True else 'objective mismatch' if r.get('final_ok') is True else 'failed',
                'quota_error':quota_error(r),
                'attempt_directory':str(Path(r['_result_path']).parent)}
            all_cases.append(detail)
            if name.startswith('full_review_'):
                previous = baseline[r['problem_id']]
                pairs.append({'run':name, 'benchmark':detail['benchmark'], 'problem_id':r['problem_id'],
                    'before_correct':previous.get('solution_correct') is True, 'after_correct':detail['objective_match'],
                    'before_solved':previous.get('final_ok') is True, 'after_solved':detail['solved'],
                    'before_objective':previous.get('final_objective'), 'after_objective':r.get('final_objective'),
                    'quota_error':quota_error(r)})
    write_csv('summary.csv',summaries,['run','method','benchmark','evaluated','total','objective_match','solved','errors','mismatches','complete','quota_errors','run_state'])
    fields = ['run','benchmark','problem_id','outcome','objective_match','solved','assigned_route','predicted_label',
              'true_label','disabled_workflow_route','held_out_type','final_objective','label_objective',
              'pipeline_stage','execution_error_type','execution_error','seconds','api_retry_count',
              'external_deadline_exceeded','notebook_source_sha256','quota_error','attempt_directory']
    write_csv('all_cases.csv',all_cases,fields)
    write_csv('unsuccessful_cases.csv',[r for r in all_cases if r['outcome']!='correct'],fields)
    if chosen:
        selected_runs = {chosen, *[control.get('method_runs', {}).get(method, f'{method}_{chosen}') for method in OLD_METHODS]}
        latest_cases = [r for r in all_cases if r['run'] in selected_runs]
        write_csv('latest_cases.csv', latest_cases, fields)
        write_csv('latest_unsuccessful_cases.csv', [r for r in latest_cases if r['outcome']!='correct'], fields)
    write_csv('paired_changes.csv',pairs,['run','benchmark','problem_id','before_correct','after_correct','before_solved','after_solved','before_objective','after_objective','quota_error'])
    (REPORT/'runtime_stats.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2))
    lines = ['# 0927 审阅、通用修复与完整验证', '',
        'Objective match：最优目标值按 rel_tol=abs_tol=1e-4 与参考答案匹配。Solved：成功得到最优解。全部失败、超时和不匹配保留在完整分母中。', '',
        '比较基线是本次审阅开始时的 full_v1：430/452 Objective match，442/452 Solved；主集为 92/101、96/101。更早的 final_101_V2 主集 95/101、99/101 属于历史记录，单独标识，未替代本轮基线。', '',
        '最新比较规则：完整 452 题的 Objective match 整体增加，允许个别数据集下降；同时保留先前 Variants ≥32/36、九个冗余列 sheet 各 ≥31/35 的目标。Solved 单独报告。全量完整通过后才派生、运行新消融和 LOTO。']
    if pause and pause.get('state') == 'saved_pending_resume':
        lines += ['', '## 已保存，等待继续', '',
            '用户要求“先保留代码和当前记录”。后续 API 运行已停止；根目录四份基线 notebook 保持本轮开始时的内容，新候选另存，尚未替换为最终版本。',
            f'- [最新候选代码]({pause["candidate"]})：通过 {pause["offline_checks"]} 项离线检查；SHA256 `{pause["candidate_sha256"]}`。',
            f'- [暂停状态及恢复规则]({pause_path})。', '',
            '最新候选在额度耗尽前完成的五组结果如下，均来自同一轮实际运行：', '',
            '| 数据集 | 基线 Objective / Solved | 候选 Objective / Solved | Objective 变化 |',
            '|---|---:|---:|---:|']
        for r in pause['completed_cohorts_without_quota_errors']:
            b = next(x for x in summaries if x['run']=='full_v1' and x['benchmark']==r['benchmark'])
            total = r['expected']
            lines.append(f"| {r['benchmark']} | {b['objective_match']}/{total} / {b['solved']}/{total} | {r['objective_match']}/{total} / {r['solved']}/{total} | {r['objective_match']-b['objective_match']:+d} |")
        lines += ['',
            '剩余六个冗余列 sheet 未完成有效验证。API 返回 credit_balance_exhausted；v5、v6 的实际额度错误和被中断的尝试原样保留，不能把这些外部失败解释为模型性能下降，也不能据此确认整体达标。Variants 的一个失败为 KeyError，属于真实执行失败，仍计入 36 题分母。',
            '恢复后应以相同候选源码启动新的一轮完整 452 题，保留当前轮次原始记录；完整验证通过后才派生、运行新消融及 LOTO。']
    resume_path = OUT/'resume_plan.json'
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        lines += ['', '## API 恢复后的运行范围', '',
            f'- [预先固定的恢复计划]({resume_path})；[API 与 embedding 检查]({OUT}/api_resume_check.json)。',
            '- 用户已更新密钥，要求运行剩余冗余列及后续消融、LOTO。六个 sheet 共 210 题整组重跑，保留先前完成的主集、Variants 和 50pct 三组；所有组使用相同冻结候选源码。',
            '- 全量汇总覆盖 452 题，来源于两个 API 批次；这不是一轮连续 452 题的新运行。各组来源及逐题原始记录哈希另行保存，旧额度失败记录保留。',
            '- 汇总经源码、模型、数据、评分核对并满足约定整体门槛后，才派生和运行新消融及 LOTO。']
    lines += ['', '## 各轮整体结果', '',
        '| 完整版本 | 完成题数 | Objective match | Solved | 相对 430 的变化 | 扩展集目标 |',
        '|---|---:|---:|---:|---:|---|']
    for name in folders:
        if not name.startswith('full_'):continue
        group = [r for r in summaries if r['run']==name]
        if not group:continue
        done = sum(r['evaluated'] for r in group); total = sum(r['total'] for r in group)
        obj = sum(r['objective_match'] for r in group); solved = sum(r['solved'] for r in group)
        complete = done==total
        full_selection = total == 452 and {r['benchmark'] for r in group} == {r['benchmark'] for r in summaries if r['run']=='full_v1'}
        blocked = statuses[name].get('state') == 'externally_blocked' or any(r['quota_errors'] for r in group)
        target = all(r['objective_match'] >= (32 if r['benchmark']=='variants' else 31) for r in group if r['benchmark']!='main')
        if blocked:
            quota = sum(r['quota_errors'] for r in group)
            interrupted = len(statuses[name].get('interrupted_attempts', []))
            lines.append(f'| {name}（已停止） | {done}/{total} 条结果；{quota} 条额度错误、{interrupted} 条中断 | {obj}/{total}（原始计数） | {solved}/{total}（原始计数） | 无有效整体比较 | 尚未确认 |')
            continue
        if not full_selection:
            lines.append(f'| {name}（部分数据集批次） | {done}/{total} | {obj}/{total} | {solved}/{total} | 待全量汇总 | 待全量汇总 |')
            continue
        lines.append(f"| {name} | {done}/{total} | {obj}/{total}{'' if complete else '（进度）'} | {solved}/{total} | {obj-430:+d}"+('' if complete else '（待完成）')+f" | {'达到' if complete and target else '未达到' if complete else '待完成'} |")
    if chosen:
        lines += ['', f'最终选择完整版本 **{chosen}**。评测后交付仅调整文件名、输出目录和说明；推理函数与提示常量与冻结评测版一致，经审计核对。', '',
            '## 最终全量与当前基线', '',
            '| 数据集 | 基线 Objective / Solved | 最终 Objective / Solved | Objective 变化 |',
            '|---|---:|---:|---:|']
        for r in [x for x in summaries if x['run']==chosen]:
            b=next(x for x in summaries if x['run']=='full_v1' and x['benchmark']==r['benchmark'])
            lines.append(f"| {r['benchmark']} | {b['objective_match']}/{r['total']} / {b['solved']}/{r['total']} | {r['objective_match']}/{r['total']} / {r['solved']}/{r['total']} | {r['objective_match']-b['objective_match']:+d} |")
        group=[r for r in summaries if r['run']==chosen]
        obj=sum(r['objective_match'] for r in group); sol=sum(r['solved'] for r in group)
        gains=sum(not r['before_correct'] and r['after_correct'] for r in pairs if r['run']==chosen)
        losses=sum(r['before_correct'] and not r['after_correct'] for r in pairs if r['run']==chosen)
        provenance = json.loads((RUNS/chosen/'run_manifest.json').read_text()).get('cohort_provenance')
        batch_note = '同一冻结源码的两个 API 批次，按预先固定的整组来源汇总；未按单题选择结果。' if provenance else '完整单轮结果，未按单题选择结果。'
        lines += ['',f'整体 Objective match：430/452 → {obj}/452，变化 {obj-430:+d} 题（{100*(obj-430)/452:+.2f} 个百分点）。Solved：442/452 → {sol}/452。逐题对照：改善 {gains} 题、退化 {losses} 题；{batch_note}', '',
            '## 最终组件移除实验', '',
            '| 方法 | 当前基线 Objective / Solved | 本轮 Objective / Solved | 相对新全量 Objective 变化 |',
            '|---|---:|---:|---:|']
        full_main=next(r for r in group if r['benchmark']=='main')
        comparisons=[]
        for method,old_name in OLD_METHODS.items():
            old=next(r for r in summaries if r['run']==old_name)
            active_run = control.get('method_runs', {}).get(method, f'{method}_{chosen}')
            new=next((r for r in summaries if r['run']==active_run),None)
            if not new:
                lines.append(f'| {method} | {old["objective_match"]}/101 / {old["solved"]}/101 | 尚未运行 | — |');continue
            valid = new['complete'] and not new['quota_errors'] and new['run_state'] != 'externally_blocked'
            delta=new['objective_match']-full_main['objective_match'] if valid else None
            if new['quota_errors'] or new['run_state'] == 'externally_blocked':
                lines.append(f"| {method} | {old['objective_match']}/101 / {old['solved']}/101 | API 额度耗尽；原始计数 {new['objective_match']}/101 / {new['solved']}/101，含 {new['quota_errors']} 条额度错误 | 无有效性能比较 |")
            else:
                lines.append(f"| {method} | {old['objective_match']}/101 / {old['solved']}/101 | {new['objective_match']}/101 / {new['solved']}/101{'' if new['complete'] else '（进度）'} | {delta if delta is not None else '待完成'} |")
            comparisons.append({'method':method,**new,'previous_objective_match':old['objective_match'],'previous_solved':old['solved'],
                'objective_delta_full':delta,'objective_percentage_point_delta_full':100*delta/101 if delta is not None else None,
                'valid_performance_comparison':valid})
        write_csv('method_comparisons.csv',comparisons,['method','run','evaluated','total','objective_match','solved','previous_objective_match','previous_solved','objective_delta_full','objective_percentage_point_delta_full','complete','quota_errors','run_state','valid_performance_comparison'])
        blocked_methods = [r for r in comparisons if r['quota_errors'] or r['run_state']=='externally_blocked']
        if blocked_methods:
            lines += ['', '**组件实验未全部完成：API 再次耗尽额度。额度错误不属于移除组件导致的性能下降，不能将含额度错误的原始计数解释为消融准确率。** 已记录的真实生成失败、截断和不匹配仍保留；恢复额度后需另开完整 101 题轮次，保持同一源码和分类缓存。各方法的启动与完成状态见上表。']
        prior_rag = inventory.get(f'rag_only_{chosen}', [])
        prior_quota = sum(quota_error(r) for r in prior_rag)
        if prior_quota:
            lines += ['', f'RAG Only 前一轮留下 {prior_quota} 条额度错误，不能用于组件性能比较。当前选用另开的完整 r2 轮次；保留前一轮全部原始记录，没有按单题替换或选择结果。']
        if len(comparisons) == 3 and all(r['valid_performance_comparison'] for r in comparisons):
            lines += ['', '三项组件实验均完成整轮 101 题，没有额度错误；所有真实失败和目标值不匹配均计入分母。与前一版本组件基线的变化、与最终全量的变化分别记录；没有为压低消融准确率额外削弱提示或执行器。', '', '### 最终各方法的失败与不匹配', '', '| 方法 | 求解失败 | 已求解但目标值不匹配 | 失败类型 |', '|---|---:|---:|---|']
            for item in comparisons:
                selected = inventory[item['run']]
                errors = Counter(r.get('execution_error_type') or 'unknown' for r in selected if r.get('final_ok') is not True)
                lines.append(f"| {item['method']} | {101-item['solved']} | {item['solved']-item['objective_match']} | {json.dumps(dict(errors),ensure_ascii=False)} |")
            loto_run=control.get('method_runs', {}).get('examples_and_route', f'examples_and_route_{chosen}')
            loto_rows=inventory[loto_run]
            fold_rows=[]
            lines += ['', '### LOTO 各类别与实际替代 route', '', '| 排除类别 | 禁用 route | Objective / Solved | 实际所选 route（题数） |', '|---|---|---:|---|']
            for label in ['TP','NRM','RA','FLP','AP','Mixture','Others']:
                fold=[r for r in loto_rows if r.get('held_out_type')==label]
                disabled=sorted({r['disabled_workflow_route'] for r in fold})
                assert len(disabled)==1
                routes=dict(Counter(r.get('assigned_route') or 'unassigned' for r in fold))
                objective=sum(r.get('solution_correct') is True for r in fold)
                solved=sum(r.get('final_ok') is True for r in fold)
                forbidden=sum(r.get('assigned_route')==disabled[0] for r in fold)
                assert forbidden==0
                fold_rows.append({'held_out_type':label,'disabled_route':disabled[0],'evaluated':len(fold),'objective_match':objective,'solved':solved,'forbidden_routes_used':forbidden,'assigned_route_counts':json.dumps(routes,ensure_ascii=False)})
                lines.append(f'| {label} | {disabled[0]} | {objective}/{len(fold)} / {solved}/{len(fold)} | {json.dumps(routes,ensure_ascii=False)} |')
            write_csv('loto_folds.csv',fold_rows,['held_out_type','disabled_route','evaluated','objective_match','solved','forbidden_routes_used','assigned_route_counts'])
    else:
        lines += ['', '**全量尚未确定；未启动新的消融及 LOTO。**']
    lines += ['', '## 改动与通用性', '',
        f'- [完整逐项代码审阅]({OUT}/review.md)：数据解析、抽取验证、标识符、提示冲突、缓存、实验边界及修复理由。',
        '- 保留 ReAct；NRM 延续既有 planned，其余 route 延续 legacy。模型快照 gpt-4.1-2025-04-14、temperature=0、top_p=1、embedding、SDK retry=1 及匹配容差不变；通用求解执行器不变。',
        '- 新提示规则均从问题本身确定：可选整数批量、双向启用关系、资格等级、按字段识别并合并分表、先版本/删除筛选后数值转换、保留已有业务键。没有题号、预设答案、特定 tenant/日期或 sheet 分支。',
        '- legacy 来源校验对比原 source、字段、值与重复行数量；发现改写、派生字段、外来行或多余重复行时记录原因和原响应，向同一建模 agent 返回原始完整证据。仍由模型按原 query 选择范围，保持 legacy/ReAct。具体使用次数见 runtime_stats.json；不表示已验证每个合法子集的语义选择。',
        '- 源码在每轮执行前冻结；中途不改代码，不根据正确率单题重跑或挑选。多项改动与模型生成波动同时存在，单次总分提高不等于已隔离每项改动的因果效应，也不能保证所有未来数据集。', '',
        '## 消融与 LOTO 边界', '',
        '以下为已审阅的派生规则；派生和运行状态见上方最终组件移除实验。' if chosen else '以下为已审阅的派生规则；本轮尚未派生或启动新的消融及 LOTO。', '',
        '- RAG Only：删除全部 route 的建模/代码参考示例；保留 CSVQA 工具、源 CSV 的完整行上下文与 legacy 抽取、NRM 计划及 Python 执行、Others 字段信息与求解时完整读取。legacy CSVQA 是 stuff chain，FAISS 用于参考示例检索。删除示例明细另见下方 CSV。',
        '- Few-shot Only：保留参考示例；Python 读取全部原始 CSV 行列作为建模 Observation，无 LLM 预筛选/摘要/改写/补值；移除 CSVQA 与当前数据抽取计划。完整原始 Observation 不传入代码生成，代码仅接收建模输出、问题及结构信息。仍用原 ReAct 解析/执行机制，当前工具列表为空。',
        '- 两份消融共享最终全量的预测分类缓存。保留相同通用数学提示与执行器；不为压低分数额外削弱提示或制造失败。',
        '- LOTO：真实类别只用于 fold；分类 RefData、建模和代码示例均移除该类别，禁用对应 route 后重新分类。分类、建模、示例检索及代码生成前检查禁用边界；禁止读取参考模型/答案辅助生成。', '',
        '## 运行与复现', '',
        f'- [全部逐题结果]({REPORT}/all_cases.csv)',
        f'- [全部失败、超时与目标值不匹配]({REPORT}/unsuccessful_cases.csv)',
        f'- [本次最终版本逐题结果]({REPORT}/latest_cases.csv)' if chosen else '',
        f'- [本次最终版本失败与不匹配]({REPORT}/latest_unsuccessful_cases.csv)' if chosen else '',
        f'- [与当前基线逐题对照]({REPORT}/paired_changes.csv)',
        f'- [SDK 重试、外部截止时间、错误类型]({REPORT}/runtime_stats.json)',
        '- 每轮运行目录保留 frozen_notebook.ipynb、run_manifest.json（源码与数据哈希）和 attempts（提示、用量、日志、原始结果、模型、Observation、代码、解与错误）；LOTO 另有逐题 fold_manifest。', '']
    from evaluate_0927_optimization import namespace
    ns=namespace(RUNS/'full_v1/frozen_notebook.ipynb')
    ref=ns['read_csv_compat'](ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv',dtype=str,keep_default_na=False)
    removed=[{'row_index':int(i),'type':r['Type'],'prompt':r['prompt'],'data_address':r['Data_address']} for i,r in ref.iterrows()]
    write_csv('rag_removed_examples.csv',removed,['row_index','type','prompt','data_address'])
    lines += [f'- [RAG Only 删除的 {len(removed)} 条参考示例明细]({REPORT}/rag_removed_examples.csv)']
    query_only=ns['read_csv_compat'](ROOT/'Large_Scale_Or_Files/RAG_Example_Others_Without_CSV.csv',dtype=str,keep_default_na=False)
    removed_query_only=[{'row_index':int(i),'type':r.get('problem type',''),'prompt':r['prompt'],'data_address':r.get('Data_address','')} for i,r in query_only.iterrows()]
    write_csv('rag_removed_query_only_examples.csv',removed_query_only,['row_index','type','prompt','data_address'])
    lines += [f'- [RAG Only 无 CSV 分支排除的 {len(removed_query_only)} 条参考示例]({REPORT}/rag_removed_query_only_examples.csv)：同时删除 INTEGER、MULTI-PERIOD FLOW、LOGIC+BINARY 三个固定示范及仅检索参考示例的 ORLM_QA；当前 101 题均有外部 CSV，未覆盖此分支。源数据 CSVQA 保留。']
    audit=OUT/'final_boundary_audit.json'
    if audit.exists():
        result=json.loads(audit.read_text())
        lines += ['',f'- [最终源码、数据与组件边界审计]({audit})',f'实际运行边界核对：{json.dumps(result["runtime"],ensure_ascii=False)}']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    print(REPORT/'report.md')


if __name__=='__main__':main()
