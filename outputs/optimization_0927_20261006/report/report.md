# 0927 最小修改与完整评测记录

正确统一使用 Objective match：Gurobi 最优目标值与参考答案按 rel_tol=abs_tol=1e-4 比较。Solved 单独统计。失败、超时和不匹配全部保留，分母使用完整题数。

修改前主数据集基准来自已经验证对应代码/数据的 final_101_V2：Objective match 95/101，Solved 99/101；修改前 Variants 和九个冗余列 sheet 在本轮重新执行。

| 运行版本 | 数据集 | 完成 | Objective match | Solved | 目标 |
|---|---|---:|---:|---:|---:|
| examples_and_route_final | main | 101/101 | 91/101 | 97/101 | — |
| few_shot_only_final | main | 101/101 | 89/101 | 92/101 | — |
| few_shot_only_final_v2 | main | 101/101 | 86/101 | 92/101 | — |
| full_before | variants | 36/36 | 28/36 | 30/36 | 32 |
| full_before | 50pct-S1 | 35/35 | 33/35 | 33/35 | 31 |
| full_before | 50pct-S2 | 35/35 | 31/35 | 35/35 | 31 |
| full_before | 50pct-S3 | 35/35 | 33/35 | 35/35 | 31 |
| full_before | 100pct-S1 | 35/35 | 32/35 | 33/35 | 31 |
| full_before | 100pct-S2 | 35/35 | 35/35 | 35/35 | 31 |
| full_before | 100pct-S3 | 35/35 | 31/35 | 33/35 | 31 |
| full_before | 200pct-S1 | 35/35 | 27/35 | 28/35 | 31 |
| full_before | 200pct-S2 | 35/35 | 32/35 | 34/35 | 31 |
| full_before | 200pct-S3 | 35/35 | 32/35 | 33/35 | 31 |
| full_v1 | main | 101/101 | 92/101 | 96/101 | — |
| full_v1 | variants | 36/36 | 34/36 | 35/36 | 32 |
| full_v1 | 50pct-S1 | 35/35 | 35/35 | 35/35 | 31 |
| full_v1 | 50pct-S2 | 35/35 | 33/35 | 34/35 | 31 |
| full_v1 | 50pct-S3 | 35/35 | 32/35 | 34/35 | 31 |
| full_v1 | 100pct-S1 | 35/35 | 33/35 | 35/35 | 31 |
| full_v1 | 100pct-S2 | 35/35 | 34/35 | 34/35 | 31 |
| full_v1 | 100pct-S3 | 35/35 | 35/35 | 35/35 | 31 |
| full_v1 | 200pct-S1 | 35/35 | 34/35 | 35/35 | 31 |
| full_v1 | 200pct-S2 | 35/35 | 34/35 | 34/35 | 31 |
| full_v1 | 200pct-S3 | 35/35 | 34/35 | 35/35 | 31 |
| rag_only_final | main | 101/101 | 83/101 | 84/101 | — |

## 最终方法与全量版本比较

两份消融复用 full_v1 的 101 个分类标签和 route（分类正确 94/101）。LOTO 每题在排除目标语义类别样本、禁用对应 route 后重新分类。

Few-shot Only 最终采用 final_v2：完整数据在建模阶段提供，代码生成函数入口只接受结构信息。这项边界修正是在运行 final_v2 之前确定的，与准确率无关。中间版 few_shot_only_final 的完整结果单独保留，未逐题选择较好结果。

| 方法 | 本轮 Objective match | 本轮 Solved | 修改前 Objective match / Solved | 相对本轮全量正确题数变化 |
|---|---:|---:|---:|---:|
| rag_only | 83/101 | 84/101 | 69/101 / 70/101 | -9 |
| few_shot_only | 86/101 | 92/101 | 87/101 / 91/101 | -6 |
| examples_and_route | 91/101 | 97/101 | 88/101 / 94/101 | -1 |

## 主集回归

历史主集 Objective match 95/101，本轮 full_v1 为 92/101（−3 题、−2.97 个百分点）；Solved 99/101 → 96/101。原先正确的 OR-027、OR-041、OR-082、OR-085 变错，原先输出截断的 OR-013 变对，其余 96 题正确性不变。这五题的 route 均未改变。

- OR-027：数据包含 Organic Fruits / Organic Staples / Organic Vegetables，本轮生成代码却用 organ 精确相等筛选，导致空集。历史版本使用 prefix/contains 筛选。
- OR-041：生成模型从整数变量改为连续变量；3912 → 4408.523076923077，Objective match 失败。题目中的 scale 用语未明确整数域，不能依据参考目标值反向指定变量类型。
- OR-082：生成代码对 pandas Series 调用 casefold，而非通过字符串访问器调用，出现 AttributeError。
- OR-085：itertuples 会转换非法 Python 字段名，生成代码仍用原列名 Unnamed: 0 取属性，出现 AttributeError。
- 多项提示词在同一版本中调整；主集对照是历史运行而非同一时刻重跑原版。以上可以确定直接失败原因，不能据此隔离每项改动的因果效应或把全部变化归因于运行波动。

## 组件边界

### RAG Only

从所有 route 的建模与代码生成提示中删除以下 15 个训练参考示例（行号从 0 起）。同时停用参考示例的检索入口；分类仍使用最终全量预测缓存。具体题干和数据路径见 rag_removed_examples.csv。

| 参考行 | 语义类别 | 建模与代码示例使用的 route |
|---|---|---|
| 0 | AP | AP |
| 1 | UFLP | FLP |
| 2 | NRM | NRM |
| 3 | Mixture | Others |
| 4 | Mixture | Others |
| 5 | Mixture | Others |
| 6 | Mixture | Others |
| 7 | Mixture | Others |
| 8 | Mixture | Others |
| 9 | Mixture | Others |
| 10 | Mixture | Others |
| 11 | Others | Others |
| 12 | Others | Others |
| 13 | RA | RA |
| 14 | TP | TP |

保留本题数据的 CSVQA Tool、CSV 行文档及其向量检索、legacy 数据检索和抽取、来源字段解析、NRM 原有 planned 数据计划及 Python 执行，以及 Others 的字段信息读取和求解时完整 CSV 读取。删除的是训练参考示例，不是全部 RAG 数据功能。

### Few-shot Only

保留各 route 的训练示例及其建模/代码参考检索。删除本题 CSVQA Tool、LLM 数据检索与抽取计划；Python 按 dtype=str、keep_default_na=False 读取本题全部 CSV 行、列和单元格，以完整 Observation 交给建模 agent。建模阶段保持 ReAct 解析器和执行器，当前任务工具列表为空。

代码生成函数仅接收建模输出、问题、结构信息和保留的训练示例；原始完整 CSV Observation 不进入该函数或其提示。完整数据仅在后续 Python 求解执行时提供。Others 同样在建模时使用完整数据、代码生成时仅使用结构信息。正常建模输出及参考示例中出现的数值不等于重新传入完整原始 CSV payload。

本轮还观察到生成波动：中间 Few-shot 版本为 89/101，最终严格边界版本为 86/101；两次运行 100/101 题的首次建模提示完全相同，正确性变化的 7 题首次建模提示也均相同。不能将这 3 题净变化解释为数据边界修正的因果效应，比较记录见 fewshot_boundary_prompt_comparison.json。最终始终使用完整 final_v2 结果。

### LOTO Examples And Route

按真实语义类别划分 fold，仅用于示例排除与禁用边界。分类 RefData 和建模/代码参考库均先删除目标语义类别，再进行 route 合并；UFLP 归一为 FLP。Mixture 和 Others 共用 Others route，因此二者 fold 都禁用 Others route 及其输出标签。分类 agent 每题在允许标签中重新分类，真实类别不用于指定替代 route。

分类、建模、参考检索和代码生成均设禁用 route 检查；参考模型和参考目标值只用于事后评价。最终运行边界审计见 final_boundary_audit.json。


## 修改范围

### v1

- shared HTTP pool and one SDK transport retry, with recorded retry events
- source-labelled JSON legacy evidence and preserve its source during parsing
- conditional/unconditional constraint guidance
- concise Others abstract plan
- non-shadowing variable names and raw CSV strings
- AP eligibility mask guidance
- explicit model-object interface guidance

改动依据案例：Variant1, Variant13, Variant17, Variant27, Variant28, Variant30, Variant32, Variant35, 50pct-S1/OR-007, 50pct-S1/OR-014.

实际局部改动说明：

- HTTP：共享连接池，将统一 SDK 最大重试从 0 改为 1，180 秒模型请求超时保持原值。修改前扩展集出现连接错误；每个版本均记录实际重试。full_v1 的 452 题实际 SDK 重试为 0，不能将其提高的正确数归因于成功重试。
- legacy 来源：行文档和抽取结果保留 source；抽取提示要求单个 JSON 数组，减少标题、Markdown 和省略号导致的解析问题。
- 约束语义：明确固定启用费用不会自动将无条件数量下限改为条件下限，依据 Variant32 的失败。
- Others：只压缩抽象计划的重复、枚举内容，保持七个章节和所有约束，依据 Variant1 的输出截断。
- 代码：添加变量容器不覆盖、quicksum 参数类型、字符串读取 CSV 后显式转数值的提示，依据 Variant28/35 等生成错误。
- AP：将不可用配对作为决策掩码处理，不能因不可用配对存在成本值而拒绝数据，依据 Variant27。
- 模型接口：要求在模块级暴露存活的 m 模型，不把 optimize() 返回值当模型。执行器本身不变。
- 框架和模式：ReAct 不变，只有原有 NRM planned，其余仍为 legacy；模型快照、采样参数、求解器指令和评分容差未修改。

## 复现与审计

- 每个运行目录包含 frozen_notebook.ipynb 和 run_manifest.json，保存代码及输入哈希、模型设置和完整题目清单。
- attempts 中保留每题 run.log、prompts.jsonl、attempt.json 和 result.json；通用版本运行器另存 usage.jsonl，LOTO 另存 fold_manifest.json。
- all_cases.csv 列出所有案例；unsuccessful_cases.csv 列出所有失败、超时及目标值不匹配；paired_changes.csv 对照每题修改前后的结果。
- SDK 对瞬时服务/网络失败的统一重试属于同一次 pipeline，事件另行记录；没有根据目标值更换结果、排除案例或重跑单题。

### 本轮边界检查

- 交付全量 notebook 的函数 AST（包括字符串常量）及提示词、route 模式常量与 full_v1 冻结评测版本一致；交付文件和输入哈希均已核对。
- 两份消融完整 101 题的分类标签、route、题干、数据路径均与最终全量分类缓存一致。
- Few-shot Only 已核对 101 份完整 Python Observation，共 289332 个本题原始 CSV 单元格；代码生成入口拒绝原始 records。输入过长导致的模型调用失败仍计入结果。
- LOTO 完整 101 题均在允许 route 上执行；七个 fold 的目标示例排除及执行前禁用检查通过。
- 各版本所有案例的失败类型、SDK 重试和外部超时见 runtime_stats.json；RAG Only 有 4 个 API 请求超时案例。

### 交付代码

- [全量](/Users/cora/Documents/GitHub/lean-llm-opt/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb)
- [RAG Only](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb)
- [Few-shot Only](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb)
- [LOTO Examples And Route](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb)

- [修改前全量原始归档](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/source_snapshots/before/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb)
- [逐题完整结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/report/all_cases.csv)
- [所有失败、超时及目标值不匹配](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/report/unsuccessful_cases.csv)
- [主集及扩展集逐题前后对照](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/report/paired_changes.csv)
- [最终组件边界审计](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/final_boundary_audit.json)
