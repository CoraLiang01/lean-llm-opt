"""Report every recorded case and paired changes from the matching historical full baseline."""
import contextlib
import io
import json
import re
from pathlib import Path
import sys

import pandas as pd
from check_0927_experiments import ROOT,NAMES,namespace

OUT=ROOT/'outputs/ablation_loto_0927'
REPORT=OUT/'report';REPORT.mkdir(parents=True,exist_ok=True)
base=namespace(ROOT/'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb')
full=base['load_records'](ROOT/'outputs/final_101_V2/automatic/results.csv')
full_by_id={r['problem_id']:r for r in full}
rows=[];details=[];boundaries=[];failures=[];paired=[];interface_errors=[]

rows.append({'method':'Full (saved baseline)','evaluated':101,'total':101,
    'solved':sum(r.get('final_ok') is True for r in full),
    'objective_match':sum(r.get('solution_correct') is True for r in full),
    'classification_correct':sum(r.get('classification_correct') is True for r in full),
    'solved_rate':sum(r.get('final_ok') is True for r in full)/101,
    'objective_accuracy':sum(r.get('solution_correct') is True for r in full)/101,
    'solved_delta_pp':0.0,'objective_delta_pp':0.0,'status':'stored baseline'})
for method,name in NAMES.items():
    ns=namespace(ROOT/name)
    root=OUT/(method+'_v1')
    records=ns['load_records'](root/'automatic/results.csv')
    recorded={r['problem_id'] for r in records}
    expected={r['problem_id'] for r in full}
    assert recorded<=expected
    summary={'method':method,'evaluated':len(records),'total':101,
        'solved':sum(r.get('final_ok') is True for r in records),
        'objective_match':sum(r.get('solution_correct') is True for r in records),
        'classification_correct':sum(r.get('classification_correct') is True for r in records)}
    summary.update(solved_rate=summary['solved']/101,objective_accuracy=summary['objective_match']/101,
        solved_delta_pp=(summary['solved']/101-rows[0]['solved_rate'])*100,
        objective_delta_pp=(summary['objective_match']/101-rows[0]['objective_accuracy'])*100,
        status='complete' if len(records)==101 else 'in progress')
    rows.append(summary)
    loss=gain=both=neither=0
    for record in records:
        previous=full_by_id[record['problem_id']]
        a=previous.get('solution_correct') is True;b=record.get('solution_correct') is True
        loss+=a and not b;gain+=not a and b;both+=a and b;neither+=not a and not b
        details.append({'method':method,**{k:record.get(k) for k in ['problem_id','true_label','predicted_label','assigned_route',
            'disabled_workflow_route','pipeline_stage','final_ok','final_objective','label_objective','solution_correct',
            'classification_correct','csvqa_mode','csvqa_status','execution_error_type','execution_error','seconds']},
            'baseline_objective_correct':a,'baseline_solved':previous.get('final_ok')})
        if record.get('execution_error'):failures.append({'method':method,'problem_id':record['problem_id'],
            'stage':record.get('pipeline_stage'),'error_type':record.get('execution_error_type'),'error':record.get('execution_error')})
        attempt=root/'attempts'/record['problem_id']
        prompts=[]
        if record.get('execution_error_type')=='AttributeError' and '__enter__' in record.get('execution_error',''):
            log=(attempt/'run.log').read_text(errors='replace')
            interface_errors.append({'method':method,'problem_id':record['problem_id'],
                'solver_optimal_logged':bool(re.search(r'Optimal solution found|Optimal objective',log)),
                'error':record['execution_error']})
        if (attempt/'prompts.jsonl').exists():
            for line in (attempt/'prompts.jsonl').read_text().splitlines():
                prompts.extend(m['content'] for group in json.loads(line)['messages'] for m in group)
        if method=='examples_and_route':
            manifest=json.loads((attempt/'fold_manifest.json').read_text())
            route=record.get('assigned_route')
            assert route is None or route in manifest['allowed_routes'],(record['problem_id'],route)
            assert record.get('predicted_label') is None or record['predicted_label'] in manifest['allowed_labels']
            boundaries.append({'method':method,'problem_id':record['problem_id'],
                'selected_route':route,'disabled_route':manifest['disabled_workflow_route'],
                'route_violation':bool(route and route==manifest['disabled_workflow_route'])})
        elif method=='few_shot_only':
            obs_path=record.get('csvqa_observation_path')
            if obs_path:
                payload=json.loads((root/'automatic'/obs_path).read_text())
                for table in payload['tables']:
                    frame=ns['read_csv_compat'](table['source'],dtype=str,keep_default_na=False)
                    actual=[r['values'] for r in table['records']]
                    assert actual==frame.to_dict('records'),record['problem_id']
                    assert table['columns']==list(frame.columns)
                raw_obs=json.dumps(payload,ensure_ascii=False)
                code_prompts=[prompt for prompt in prompts if
                    'Mathematical Optimization Model:' in prompt or 'Your task is to strictly follow the User Query' in prompt]
                assert all(raw_obs not in prompt and '\"source_row\"' not in prompt and 'Complete LEGACY_RECORDS:' not in prompt
                           and 'Complete CSVQA_DATA:' not in prompt for prompt in code_prompts),record['problem_id']
                boundaries.append({'method':method,'problem_id':record['problem_id'],
                    'observation_cells_exact':True,'complete_data_in_code_prompt':False})
    paired.append({'method':method,'evaluated':len(records),'both_objective_correct':both,
                   'baseline_correct_new_incorrect':loss,'baseline_incorrect_new_correct':gain,'both_incorrect':neither})

pd.DataFrame(rows).to_csv(REPORT/'comparison.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(details).to_csv(REPORT/'case_comparison.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(paired).to_csv(REPORT/'paired_objective_changes.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(interface_errors).to_csv(REPORT/'model_interface_errors.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(failures).to_csv(REPORT/'failures.csv',index=False,encoding='utf-8-sig')
error_counts=pd.DataFrame(failures).groupby(['method','error_type']).size().rename('cases').reset_index() if failures else pd.DataFrame(columns=['method','error_type','cases'])
error_counts.to_csv(REPORT/'failure_counts.csv',index=False,encoding='utf-8-sig')
(REPORT/'runtime_boundary_audit.json').write_text(json.dumps(boundaries,indent=2))
refs=base['read_csv_compat'](ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv',dtype=str,keep_default_na=False)
ref_rows=[{'row':int(i),'semantic_type':base['normalize_problem_class'](r['Type']),
           'prompt':r['prompt'],'removed_fields':['prompt','Required Data','Label','Code']} for i,r in refs.iterrows()]
(REPORT/'rag_only_removed_examples.json').write_text(json.dumps(ref_rows,ensure_ascii=False,indent=2))

labels={'rag_only':'RAG Only','few_shot_only':'Few-shot Only','examples_and_route':'LOTO Examples And Route'}
lines=['# 0927 全量代码、消融与 LOTO 实验','',
'基准为 `LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb`，保存结果位于 `outputs/final_101_V2`。按历史文件路径重建的代码/数据缓存指纹与全部 **101/101** 已保存全量结果一致。指纹包含定义代码、模型及 embedding 配置、参考 CSV 和本题 CSV 原始字节；忽略 notebook 输出和实验开关。证据见 `../baseline_provenance.json`。','',
'本次每个实验使用一个预先确定的实现，每题一次完整流程；所有正常解、错误和不匹配结果均计入 101 题分母。完整数据、生成模型、程序、解、模型请求及运行日志均保留。表中的全量结果是已保存的匹配基准，本轮不重复计算全量结果。','',
'## 结果与全量相比','',
'| 方法 | 完成题数 | Solved | Objective match | Solved 变化 | Objective 变化 |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:
    label=labels.get(r['method'],r['method'])
    lines.append(f"| {label} | {r['evaluated']}/101 | {r['solved']}/101 ({r['solved_rate']:.2%}) | {r['objective_match']}/101 ({r['objective_accuracy']:.2%}) | {r['solved_delta_pp']:+.2f} 个百分点 | {r['objective_delta_pp']:+.2f} 个百分点 |")
if not all(r['evaluated']==101 for r in rows):lines+=['','**实验仍在运行，当前表为进度，不是最终准确率。**']
lines += ['', 'Solved 表示原有通用执行机制成功接收生成程序提供的模型对象并确认 Gurobi 最优解；Objective match 表示成功提取的最优目标值在 `rel_tol=abs_tol=1e-4` 下匹配参考答案，它是这里的主要正确率指标。两份消融复用全量的标签/route，因此分类正确率保持 95/101（94.06%）。LOTO 禁止真实 held-out 标签及其 route；其分类对真实标签的准确率必然为 0，不能把该值作为求解性能指标。','',
'## 失败与计分边界','',
'上下文超限、API 超时、生成程序错误和求解错误均如实作为失败计入 101 题分母；不通过截断完整 Observation 绕过上下文限制，也不排除这些案例。逐类计数见 failure_counts.csv，具体原因见 failures.csv。','',
'部分 RAG Only 程序的日志显示求解器已完成优化，但生成函数没有按提示要求把 Gurobi 模型对象留在 m 或 model（例如把没有 return 的函数结果赋给 m）。原有 execute_code 因此报 AttributeError: __enter__。这些计为程序接口/结果提取失败，不能据此推断数学模型一定错误。逐题证据见 model_interface_errors.csv；主表使用相同的原有执行机制计分，没有为某一个方法单独修复或补取目标值。','',
'## 保持一致的设置','',
'- 相同的 101 道题目和 CSV 数据；相同 `gpt-4.1-2025-04-14`、temperature=0、top_p=1、n=1、SDK max_retries=0。',
'- 基准仅 NRM 使用 planned CSVQA，其余 route 的 legacy 行为在 RAG Only 和 LOTO 中保留。',
'- 相同的 `_source_candidate`、`execute_code`、求解提示、最优状态判断和目标值 scorer。CSV code 提示的 MIPGap=1e-4 保持不变；Others 使用基准本身的原有代码提示。',
'- 完整运行设置、代码快照及所有输入哈希在各实验的 `run_manifest.json`；新增机制只是记录模型请求、计时与一次运行的结果。外部案例期限统一为 1800 秒，超时如实记录。','',
'## RAG Only：删除与保留','',
'删除的是 **建模和代码生成提示中的示例**，不是本题数据的检索流程。删除清单（RAG_Examples_All.csv，零起始行号）：','',
'| 行号 | 类型 | 移除用途 |','|---|---|---|',
'| 0 | AP | 建模示例及 Gurobi 代码示例 |',
'| 1 | UFLP → FLP | 建模示例及 Gurobi 代码示例 |',
'| 2 | NRM | 建模示例及 Gurobi 代码示例 |',
'| 3–10 | Mixture | Others 抽象计划及代码示例 |',
'| 11–12 | Others | 抽象计划及代码示例 |',
'| 13 | RA | 建模示例及 Gurobi 代码示例 |',
'| 14 | TP | 建模示例及 Gurobi 代码示例 |','',
'对应示例的 prompt、Required Data、Label 和 Code 均不进入这些提示。Query-only 路径中的参考示例与三个固定建模演示也删除。完整逐条清单见 `rag_only_removed_examples.json`。分类演示不删除，分类只复用全量结果，不重新调用分类 agent。','',
'保留：CSVQA Tool；`load_csv_documents` 的 source-order CSV 文档；legacy LLM 证据读取；NRM 的 `_ask_extraction_plan`、`_execute_plan`、校验和原有 full-data fallback；Others 的 Python CSV schema preview、全文件统计、题目名称匹配证据和运行时完整文件读取。提供 few-shot 的 route 示例检索停止；分类 RefData 的信息通过已经生成的分类结果保留。','',
'## Few-shot Only：数据边界','',
'保留 NRM/RA/TP/AP/FLP 原有 few-shot、Others/Mixture 的抽象计划与代码示例，以及原有示例选择与 embedding 设置。移除当前题目 CSVQA Tool、LLM CSV evidence 请求、NRM 抽取 planner/executor/fallback。Python 以字符串读取完整 CSV，每个原始行/列/单元格都进入 Observation；不做过滤、截断、LLM 筛选、LLM 摘要、数值重写或补造。','',
'完整原始 Observation 只进入建模请求。代码生成接收建模结果（可含模型所需的数值参数）及示例，不附带完整原始表/records。NRM 的代码请求仅增加无单元格值的表结构，完整 CSVQA_DATA 由 Python 在执行时绑定。Others 的代码请求仅收到 schema 元数据与 abstract plan，程序在运行时读取完整文件。运行后的逐单元格及请求审计见 `runtime_boundary_audit.json`。','',
'## LOTO Examples And Route','',
'每个目标语义类型的样本先从分类 RefData 与建模/代码参考库排除，再做 route 归一化；固定分类演示及 query-only 示例遵循同样排除规则。Mixture 与 Others 共用 Others route，所以这两个 fold 均禁用 Others route，且禁用两个对应输出标签。','',
'分类 agent 按当前题目重新分类，只能返回允许标签；被禁止的输出直接记为错误，不根据 gold 标签/参考答案指定替代 route。所有 route 在调用建模和代码生成前检查。真实类型仅用于划分评价 fold，参考目标值仅用于事后 scorer，Label-model 不进入模型输入。','',
'为了改善跨类型泛化，建模阶段加入通用约束：route 仅提供示例风格，当前题目和数据决定数学结构；不得强套参考 route 的变量、约束或额外机制。空示例库返回空列表，不填入被移除类别。没有根据每题参考答案修复程序或重新选择结果。','',
'执行前的七-fold 禁用 route 测试见 `../boundary_checks.json`，运行时审计见 `runtime_boundary_audit.json`。','',
'## Variants 与冗余列入口','',
'LOTO notebook 已参考 `_1006.ipynb` 加入独立 `RUN_VARIANTS` 和 `RUN_REDUNDANT_COLUMNS` 开关。数据及路径检查通过：Variants 36 题；50pct、100pct、200pct 各 S1/S2/S3，共九个 sheet，每个 35 题。入口调用同一个 LOTO runner，保持分类重新计算、示例排除和禁用 route 规则。本次结果表只包含用户要求实际比较的三份 101 题主实验；额外数据集入口的检查不算模型评测结果。','',
'## 结果文件','',
'- `comparison.csv`：相对全量的计数、正确率和百分点变化。',
'- `case_comparison.csv`：全部案例及所选 route、错误、目标值和与基准的逐题对照。',
'- `paired_objective_changes.csv`：全量正确→实验错误和全量错误→实验正确的案例计数。',
'- `failures.csv` 与 `failure_counts.csv`：完整失败原因及逐类计数。',
'- `../{method}_v1/automatic/`：基准格式的 results.csv、逐题模型/代码/解与汇总表。',
'- `../{method}_v1/attempts/`：每题 run.log、prompts.jsonl、result.json；LOTO 还有 fold_manifest.json。','']
(REPORT/'report.md').write_text('\n'.join(lines))
print(pd.DataFrame(rows).to_string(index=False))
print('Report:',REPORT/'report.md')
