# 0927 审阅、通用修复与完整验证

Objective match：最优目标值按 rel_tol=abs_tol=1e-4 与参考答案匹配。Solved：成功得到最优解。全部失败、超时和不匹配保留在完整分母中。

比较基线是本次审阅开始时的 full_v1：430/452 Objective match，442/452 Solved；主集为 92/101、96/101。更早的 final_101_V2 主集 95/101、99/101 属于历史记录，单独标识，未替代本轮基线。

最新比较规则：完整 452 题的 Objective match 整体增加，允许个别数据集下降；同时保留先前 Variants ≥32/36、九个冗余列 sheet 各 ≥31/35 的目标。Solved 单独报告。全量完整通过后才派生、运行新消融和 LOTO。

## API 恢复后的运行范围

- [预先固定的恢复计划](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/resume_plan.json)；[API 与 embedding 检查](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/api_resume_check.json)。
- 用户已更新密钥，要求运行剩余冗余列及后续消融、LOTO。六个 sheet 共 210 题整组重跑，保留先前完成的主集、Variants 和 50pct 三组；所有组使用相同冻结候选源码。
- 全量汇总覆盖 452 题，来源于两个 API 批次；这不是一轮连续 452 题的新运行。各组来源及逐题原始记录哈希另行保存，旧额度失败记录保留。
- 汇总经源码、模型、数据、评分核对并满足约定整体门槛后，才派生和运行新消融及 LOTO。

## 各轮整体结果

| 完整版本 | 完成题数 | Objective match | Solved | 相对 430 的变化 | 扩展集目标 |
|---|---:|---:|---:|---:|---|
| full_v1 | 452/452 | 430/452 | 442/452 | +0 | 达到 |
| full_review_v2 | 452/452 | 433/452 | 443/452 | +3 | 未达到 |
| full_review_v3 | 452/452 | 431/452 | 443/452 | +1 | 未达到 |
| full_review_v4 | 452/452 | 437/452 | 444/452 | +7 | 未达到 |
| full_review_v5（已停止） | 452/452 条结果；23 条额度错误、0 条中断 | 418/452（原始计数） | 425/452（原始计数） | 无有效整体比较 | 尚未确认 |
| full_review_v6（已停止） | 451/452 条结果；199 条额度错误、1 条中断 | 246/452（原始计数） | 251/452（原始计数） | 无有效整体比较 | 尚未确认 |
| full_review_v6_completed | 452/452 | 445/452 | 450/452 | +15 | 达到 |
| full_review_v6_remaining（部分数据集批次） | 210/210 | 209/210 | 209/210 | 待全量汇总 | 待全量汇总 |

最终选择完整版本 **full_review_v6_completed**。评测后交付仅调整文件名、输出目录和说明；推理函数与提示常量与冻结评测版一致，经审计核对。

## 最终全量与当前基线

| 数据集 | 基线 Objective / Solved | 最终 Objective / Solved | Objective 变化 |
|---|---:|---:|---:|
| main | 92/101 / 96/101 | 97/101 / 101/101 | +5 |
| variants | 34/36 / 35/36 | 35/36 / 35/36 | +1 |
| 50pct-S1 | 35/35 / 35/35 | 34/35 / 35/35 | -1 |
| 50pct-S2 | 33/35 / 34/35 | 35/35 / 35/35 | +2 |
| 50pct-S3 | 32/35 / 34/35 | 35/35 / 35/35 | +3 |
| 100pct-S1 | 33/35 / 35/35 | 35/35 / 35/35 | +2 |
| 100pct-S2 | 34/35 / 34/35 | 35/35 / 35/35 | +1 |
| 100pct-S3 | 35/35 / 35/35 | 34/35 / 34/35 | -1 |
| 200pct-S1 | 34/35 / 35/35 | 35/35 / 35/35 | +1 |
| 200pct-S2 | 34/35 / 34/35 | 35/35 / 35/35 | +1 |
| 200pct-S3 | 34/35 / 35/35 | 35/35 / 35/35 | +1 |

整体 Objective match：430/452 → 445/452，变化 +15 题（+3.32 个百分点）。Solved：442/452 → 450/452。逐题对照：改善 19 题、退化 4 题；同一冻结源码的两个 API 批次，按预先固定的整组来源汇总；未按单题选择结果。

## 最终组件移除实验

| 方法 | 当前基线 Objective / Solved | 本轮 Objective / Solved | 相对新全量 Objective 变化 |
|---|---:|---:|---:|
| rag_only | 83/101 / 84/101 | 85/101 / 91/101 | -12 |
| few_shot_only | 86/101 / 92/101 | 92/101 / 96/101 | -5 |
| examples_and_route | 91/101 / 97/101 | 89/101 / 98/101 | -8 |

RAG Only 前一轮留下 61 条额度错误，不能用于组件性能比较。当前选用另开的完整 r2 轮次；保留前一轮全部原始记录，没有按单题替换或选择结果。

三项组件实验均完成整轮 101 题，没有额度错误；所有真实失败和目标值不匹配均计入分母。与前一版本组件基线的变化、与最终全量的变化分别记录；没有为压低消融准确率额外削弱提示或执行器。

### 最终各方法的失败与不匹配

| 方法 | 求解失败 | 已求解但目标值不匹配 | 失败类型 |
|---|---:|---:|---|
| rag_only | 10 | 6 | {"ModelOutputTruncated": 5, "ValueError": 1, "TypeError": 2, "KeyError": 1, "SyntaxError": 1} |
| few_shot_only | 5 | 4 | {"KeyError": 1, "BadRequestError": 1, "ValueError": 3} |
| examples_and_route | 3 | 9 | {"BadRequestError": 1, "APITimeoutError": 1, "KeyError": 1} |

### LOTO 各类别与实际替代 route

| 排除类别 | 禁用 route | Objective / Solved | 实际所选 route（题数） |
|---|---|---:|---|
| TP | TP | 8/9 / 8/9 | {"Others": 8, "RA": 1} |
| NRM | NRM | 22/25 / 24/25 | {"RA": 25} |
| RA | RA | 21/22 / 22/22 | {"Others": 22} |
| FLP | FLP | 14/14 / 14/14 | {"Others": 14} |
| AP | AP | 5/5 / 5/5 | {"Others": 5} |
| Mixture | Others | 12/18 / 17/18 | {"RA": 14, "TP": 3, "AP": 1} |
| Others | Others | 7/8 / 8/8 | {"AP": 2, "TP": 1, "RA": 5} |

## 改动与通用性

- [完整逐项代码审阅](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/review.md)：数据解析、抽取验证、标识符、提示冲突、缓存、实验边界及修复理由。
- 保留 ReAct；NRM 延续既有 planned，其余 route 延续 legacy。模型快照 gpt-4.1-2025-04-14、temperature=0、top_p=1、embedding、SDK retry=1 及匹配容差不变；通用求解执行器不变。
- 新提示规则均从问题本身确定：可选整数批量、双向启用关系、资格等级、按字段识别并合并分表、先版本/删除筛选后数值转换、保留已有业务键。没有题号、预设答案、特定 tenant/日期或 sheet 分支。
- legacy 来源校验对比原 source、字段、值与重复行数量；发现改写、派生字段、外来行或多余重复行时记录原因和原响应，向同一建模 agent 返回原始完整证据。仍由模型按原 query 选择范围，保持 legacy/ReAct。具体使用次数见 runtime_stats.json；不表示已验证每个合法子集的语义选择。
- 源码在每轮执行前冻结；中途不改代码，不根据正确率单题重跑或挑选。多项改动与模型生成波动同时存在，单次总分提高不等于已隔离每项改动的因果效应，也不能保证所有未来数据集。

## 消融与 LOTO 边界

以下为已审阅的派生规则；派生和运行状态见上方最终组件移除实验。

- RAG Only：删除全部 route 的建模/代码参考示例；保留 CSVQA 工具、源 CSV 的完整行上下文与 legacy 抽取、NRM 计划及 Python 执行、Others 字段信息与求解时完整读取。legacy CSVQA 是 stuff chain，FAISS 用于参考示例检索。删除示例明细另见下方 CSV。
- Few-shot Only：保留参考示例；Python 读取全部原始 CSV 行列作为建模 Observation，无 LLM 预筛选/摘要/改写/补值；移除 CSVQA 与当前数据抽取计划。完整原始 Observation 不传入代码生成，代码仅接收建模输出、问题及结构信息。仍用原 ReAct 解析/执行机制，当前工具列表为空。
- 两份消融共享最终全量的预测分类缓存。保留相同通用数学提示与执行器；不为压低分数额外削弱提示或制造失败。
- LOTO：真实类别只用于 fold；分类 RefData、建模和代码示例均移除该类别，禁用对应 route 后重新分类。分类、建模、示例检索及代码生成前检查禁用边界；禁止读取参考模型/答案辅助生成。

## 运行与复现

- [全部逐题结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/all_cases.csv)
- [全部失败、超时与目标值不匹配](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/unsuccessful_cases.csv)
- [本次最终版本逐题结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/latest_cases.csv)
- [本次最终版本失败与不匹配](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/latest_unsuccessful_cases.csv)
- [与当前基线逐题对照](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/paired_changes.csv)
- [SDK 重试、外部截止时间、错误类型](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/runtime_stats.json)
- 每轮运行目录保留 frozen_notebook.ipynb、run_manifest.json（源码与数据哈希）和 attempts（提示、用量、日志、原始结果、模型、Observation、代码、解与错误）；LOTO 另有逐题 fold_manifest。

- [RAG Only 删除的 15 条参考示例明细](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_examples.csv)
- [RAG Only 无 CSV 分支排除的 79 条参考示例](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_query_only_examples.csv)：同时删除 INTEGER、MULTI-PERIOD FLOW、LOGIC+BINARY 三个固定示范及仅检索参考示例的 ORLM_QA；当前 101 题均有外部 CSV，未覆盖此分支。源数据 CSVQA 保留。

- [最终源码、数据与组件边界审计](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/final_boundary_audit.json)
实际运行边界核对：{"rag_only": {"evaluated": 101, "forbidden_routes_used": 0, "full_observations": 0, "verified_csv_cells": 0}, "few_shot_only": {"evaluated": 101, "forbidden_routes_used": 0, "full_observations": 101, "verified_csv_cells": 289332}, "examples_and_route": {"evaluated": 101, "forbidden_routes_used": 0, "full_observations": 0, "verified_csv_cells": 0}}
