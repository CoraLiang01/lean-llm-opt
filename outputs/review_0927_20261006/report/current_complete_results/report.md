# 当前全部有效结果统计与实现说明

统计基于已有完整运行记录；本次不新增模型调用。Few-shot Only 使用 NRM 改为数值 Formulation 后的新完整 101 题结果（88/101），不混用旧 92/101。

## 统计口径

- Objective match：程序求解成功后，真实 Gurobi 模型的最优目标值与参考值在 rel_tol=abs_tol=1e-4 内匹配。
- Solved：程序执行成功且 Gurobi 状态为 OPTIMAL。API 错误、输出截断、超时、执行错误均保留。
- 类别表按 true_label（真实类别）统计；Mixture 与 Others 是两个统计类别，均通常映射至 Others 执行 route。LOTO 会重新分类，因此执行 route 与真实类别通常不同。
- 模型为 gpt-4.1-2025-04-14，temperature=0、top_p=1、n=1；embedding 为 text-embedding-ada-002；SDK 重试 1 次，NRM 额外截断重试关闭；求解提示 MIPGap=1e-4，单题外部截止 1800 秒。

## 1. 总体结果

| 方法 / 数据集 | Objective match | Solved | 失败 | 已求解但不匹配 |
|---|---:|---:|---:|---:|
| Full / main | 97/101（96.04%） | 101/101（100.00%） | 0 | 4 |
| Full / variants | 35/36（97.22%） | 35/36（97.22%） | 1 | 0 |
| Full / 50pct-S1 | 34/35（97.14%） | 35/35（100.00%） | 0 | 1 |
| Full / 50pct-S2 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 50pct-S3 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 100pct-S1 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 100pct-S2 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 100pct-S3 | 34/35（97.14%） | 34/35（97.14%） | 1 | 0 |
| Full / 200pct-S1 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 200pct-S2 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| Full / 200pct-S3 | 35/35（100.00%） | 35/35（100.00%） | 0 | 0 |
| RAG Only / main | 85/101（84.16%） | 91/101（90.10%） | 10 | 6 |
| Few-shot Only / main | 88/101（87.13%） | 94/101（93.07%） | 7 | 6 |
| LOTO Examples And Route / main | 89/101（88.12%） | 98/101（97.03%） | 3 | 9 |

全量九个冗余列 sheet 共 313/315 Objective match、314/315 Solved；全量全部数据合计 445/452 Objective match、450/452 Solved。Variants 35/36 ≥32；九个 sheet 每组 34 或 35 题正确，均达到 ≥31 且 ≥32。

三项组件实验当前只完整评测了 101 主集，未完整评测 Variants 或九个冗余列 sheet，不以全量分数代替这些方法的扩展集分数。

## 2. 101 主集：各类别

| 类别 | 题数 | 全量 Objective / Solved | RAG Only Objective / Solved | Few-shot Only Objective / Solved | LOTO Objective / Solved |
|---|---:|---:|---:|---:|---:|
| TP | 9 | 9 / 9 | 8 / 9 | 9 / 9 | 8 / 8 |
| NRM | 25 | 24 / 25 | 17 / 18 | 20 / 22 | 22 / 24 |
| RA | 22 | 22 / 22 | 21 / 22 | 22 / 22 | 21 / 22 |
| FLP | 14 | 14 / 14 | 13 / 13 | 14 / 14 | 14 / 14 |
| AP | 5 | 5 / 5 | 4 / 5 | 5 / 5 | 5 / 5 |
| Mixture | 18 | 16 / 18 | 15 / 16 | 12 / 15 | 12 / 17 |
| Others | 8 | 7 / 8 | 7 / 8 | 6 / 7 | 7 / 8 |

表中两数是正确题数与 Solved 题数，不是准确率分数。RAG/Few-shot/LOTO 相对全量 Objective match 分别少 12/9/8 题，分别下降 11.88/8.91/7.92 个百分点。

Few-shot Only 相比修改前 92/101 Objective、96/101 Solved，当前 88/101、94/101；NRM 23→20，Mixture 13→12，其余类别 Objective 总数不变。其他 route 也重新生成，单次总变化不能全部归因于 NRM。

## 3. 全量 Variants：各类别

| 类别 | 题数 | Objective match | Solved |
|---|---:|---:|---:|
| TP | 0 | —（无该类别题目） | — |
| NRM | 0 | —（无该类别题目） | — |
| RA | 0 | —（无该类别题目） | — |
| FLP | 3 | 3/3 | 3/3 |
| AP | 3 | 3/3 | 3/3 |
| Mixture | 17 | 16/17 | 16/17 |
| Others | 13 | 13/13 | 13/13 |

Variant11（Mixture、Others route）发生 KeyError，其余 35 题均 Solved 且 Objective match。

## 4. 全量九个冗余列 sheet：各类别

每组分布相同：TP 6、NRM 0、RA 11、FLP 7、AP 4、Mixture 4、Others 3，共 35 题。NRM 没有题目，不能据此推断 NRM 对冗余列的鲁棒性。

### Objective match

| Sheet | TP | RA | FLP | AP | Mixture | Others | 总计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50pct-S1 | 6/6 | 11/11 | 6/7 | 4/4 | 4/4 | 3/3 | 34/35 |
| 50pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 50pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S1 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 2/3 | 34/35 |
| 200pct-S1 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 200pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 200pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |

### Solved

| Sheet | TP | RA | FLP | AP | Mixture | Others | 总计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50pct-S1 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 50pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 50pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S1 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 100pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 2/3 | 34/35 |
| 200pct-S1 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 200pct-S2 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |
| 200pct-S3 | 6/6 | 11/11 | 7/7 | 4/4 | 4/4 | 3/3 | 35/35 |

两项未匹配：50pct-S1/OR-028（FLP）已 Solved，目标 10188 vs 10419；100pct-S3/OR-033（Others）因 KeyError 未 Solved。

## 5. 结果存储路径

### 原始完整运行目录

#### Full

运行目录：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed`

[逐题 results.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/automatic/results.csv) · [原运行 summary.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/automatic/summary.csv) · [运行配置与来源 run_manifest.json](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/run_manifest.json) · [实际评测冻结 notebook](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/frozen_notebook.ipynb)

生成模型、代码、数据 Observation、抽取 trace、解：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/automatic/results_cases/`。CSV 中的 *_path 相对于该轮 automatic/ 目录。
逐题请求、输出和日志：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/attempts/<problem_id>/`。
#### RAG Only

运行目录：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2`

[逐题 results.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/automatic/results.csv) · [原运行 summary.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/automatic/summary.csv) · [运行配置与来源 run_manifest.json](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/run_manifest.json) · [实际评测冻结 notebook](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/frozen_notebook.ipynb)

生成模型、代码、数据 Observation、抽取 trace、解：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/automatic/results_cases/`。CSV 中的 *_path 相对于该轮 automatic/ 目录。
逐题请求、输出和日志：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/rag_only_full_review_v6_completed_r2/attempts/<problem_id>/`。
#### Few-shot Only

运行目录：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1`

[逐题 results.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/automatic/results.csv) · [原运行 summary.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/automatic/summary.csv) · [运行配置与来源 run_manifest.json](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/run_manifest.json) · [实际评测冻结 notebook](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/frozen_notebook.ipynb)

生成模型、代码、数据 Observation、抽取 trace、解：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/automatic/results_cases/`。CSV 中的 *_path 相对于该轮 automatic/ 目录。
逐题请求、输出和日志：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/attempts/<problem_id>/`。
#### LOTO Examples And Route

运行目录：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2`

[逐题 results.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/automatic/results.csv) · [原运行 summary.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/automatic/summary.csv) · [运行配置与来源 run_manifest.json](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/run_manifest.json) · [实际评测冻结 notebook](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/frozen_notebook.ipynb)

生成模型、代码、数据 Observation、抽取 trace、解：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/automatic/results_cases/`。CSV 中的 *_path 相对于该轮 automatic/ 目录。
逐题请求、输出和日志：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/examples_and_route_full_review_v6_completed_r2/attempts/<problem_id>/`。

### 全量两批次的来源

全量汇总是同一冻结代码在两批 API 运行上的整组覆盖，不是新的一次连续 452 题运行。主集、Variants、50pct 三组来自第一批；100pct、200pct 六组来自第二批。没有逐题选择最优结果。

第一批：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6`
第二批：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_remaining`

合并目录 attempts/<problem_id>/record_origin.json 保存对应来源。原始 run.log、prompts.jsonl、usage.jsonl 位于对应来源批次 attempts/<problem_id>/。

### 本次统一导出

目录：`/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results`

[总体统计 CSV](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/method_dataset_summary.csv) · [每方法、数据集、真实类别统计 CSV](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/method_dataset_class_summary.csv) · [按实际执行 route 统计 CSV](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/method_dataset_route_summary.csv)

单独数据集逐题导出：

- [Full / main](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_main_cases.csv)
- [Full / variants](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_variants_cases.csv)
- [Full / 50pct-S1](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_50pct-S1_cases.csv)
- [Full / 50pct-S2](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_50pct-S2_cases.csv)
- [Full / 50pct-S3](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_50pct-S3_cases.csv)
- [Full / 100pct-S1](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_100pct-S1_cases.csv)
- [Full / 100pct-S2](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_100pct-S2_cases.csv)
- [Full / 100pct-S3](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_100pct-S3_cases.csv)
- [Full / 200pct-S1](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_200pct-S1_cases.csv)
- [Full / 200pct-S2](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_200pct-S2_cases.csv)
- [Full / 200pct-S3](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/full_200pct-S3_cases.csv)
- [RAG Only / main](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/rag_only_main_cases.csv)
- [Few-shot Only / main](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/few_shot_only_main_cases.csv)
- [LOTO Examples And Route / main](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/loto_main_cases.csv)

[全部 755 条记录](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/all_cases.csv) · [全部 48 条失败/不匹配记录](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/unsuccessful_cases.csv) · [完整路径映射 JSON](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/result_paths.json) · [代码来源核对 JSON](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/source_integrity.json)

## 6. 当前代码如何实现

### 6.1 全量代码

[当前全量 notebook](/Users/cora/Documents/GitHub/lean-llm-opt/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb)

共同流程为：本题问题 → 分类 ReAct + FileQA → 所选 route 建模 → 生成 Python/Gurobi code → 执行真实模型 → 事后评分。

分类参考库是 Large_Scale_Or_Files/RAG_Examples_All.csv 的 prompt + Type 字段，FAISS 向量检索前五条。分类 agent 必须恰好调用一次 FileQA，返回 TP/NRM/RA/FLP/AP/Mixture/Others 之一；Mixture/Others 通常进入 Others 执行 route。它不使用真实类别或参考目标值决定 route。

建模与代码参考示例使用同一库中的训练参考问题、模型和代码。当前 15 条：AP 1、FLP 1、NRM 1、RA 1、TP 1、Mixture 8、Others 2。参考示例只能提供结构与编程模式，不能复制其数值、维度或标识符到本题。

| Route | 数据阶段 | Formulation / Plan | 代码与执行 |
|---|---|---|---|
| NRM（网络收益管理） | planned CSVQA：LLM 根据本题及表信息提出 JSON 表/列/筛选/映射计划；Python 校验并执行，非法计划保留错误并回退完整原始数据 | ReAct 调用一次 CSVQA；最多 1 个建模示例；输出符号模型和精确 Data Mapping | 数学模型、原问题、完整验证 payload 与最多 2 个代码示例进入代码 LLM；之后 Python 注入 CSVQA_DATA，生成代码从该变量取系数，不再读取文件 |
| RA（资源分配） | legacy CSVQA：全部源行作为上下文，由 LLM 返回本题相关来源行，再由 Python 校验来源、字段、原值和重复次数 | ReAct 调用一次 CSVQA；最多 3 个建模示例（库中只有 1 个）；数值模型区分全局活动与各资源独立分配 | 代码 LLM 获得完整 LEGACY_RECORDS 和模型；Python 把记录绑定到代码运行环境，从记录读取数据，禁止外部 CSV 读取和复制猜测系数 |
| TP（运输） | legacy CSVQA，与 RA 相同来源校验 | 至少一次 CSVQA；最多 3 个建模示例（库中只有 1 个）；完整供需、成本、标识符和数值模型 | 模型与原问题、最多 2 个代码示例生成自包含 code，系数写入程序，无原始数据注入 |
| AP（指派） | legacy CSVQA，保留成本、资格、等级与可用性字段 | 至少一次 CSVQA；最多 1 个建模示例；数值模型明确指派约束和本题资格顺序 | 依据数值模型生成自包含 code，不能把存在报价当作允许指派 |
| FLP（设施选址） | legacy CSVQA，保留设施/客户 ID、固定成本、容量、需求、运输矩阵和轴顺序 | 至少一次 CSVQA；最多 1 个建模示例；数值模型保留设施启用与运输约束 | 依据数值模型生成自包含 code，禁止默认补零、猜测转置或引入不存在的商品轴 |
| Others（其他/混合结构） | Python 读取 CSV，提供字段、类型、前 10 行、完整表统计和问题匹配证据 | 最多 3 个示例；第一个 LLMChain 生成抽象建模计划 | 第二个 LLMChain 获得计划、概要、问题、路径、代码示例；生成程序运行时读取完整 CSV 并按题筛选、连接、建模 |

框架边界：分类及 NRM/RA/TP/AP/FLP 建模使用 ReAct；带 CSV 的 Others 继承两个 LLMChain，不是 CSVQA ReAct。无 CSV 的 Others 保留原问题 + ORLM_QA 参考检索 + ReAct 建模分支；本次主集所有题均有外部 CSV，因此没有验证该无 CSV 分支。

legacy 来源校验拒绝改值、伪造来源、外来字段、额外重复及派生行；失败保存原因并回退完整源记录。它只能检验返回行的忠实性，不能证明子集语义正确或数据已全覆盖。NRM planned 校验增加过滤证据、类型、标量/向量/矩阵、文件与轴覆盖检查，计划失败时保留完整原始数据作为证据，筛选仍由本题定义。

共用提示保留目标方向和常数、条件/无条件界、整数/连续域、可选批量、双向启用逻辑、资格规则、单位和业务键映射；针对 pandas、Gurobi 接口和矩阵索引进行局部修复。执行器读取顶层活模型 m（兼容 model），只接受 OPTIMAL；没有通过打印一个答案来评分。

### 6.2 RAG Only

[RAG Only notebook](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb)

分类复用最终全量 101 题预测缓存，只复用预测类别与 route；不使用真实类别指定路由。建模与代码示例检索函数返回空列表，排除 15 条 CSV 训练参考及其模型/代码；无 CSV 分支另排除 79 条参考及 INTEGER、MULTI-PERIOD FLOW、LOGIC+BINARY 三个固定示范。分类缓存属于既定实验边界，不从分类中移除示例贡献。

保留本题数据访问：NRM planned CSVQA + Python 校验抽取 + 回退；RA/TP/AP/FLP legacy CSVQA + 来源校验 + 回退；Others 概要、抽象计划与执行时完整读 CSV。NRM 的 CSVQA_DATA、RA 的 LEGACY_RECORDS 数据机制保留。通用数学要求、模型与求解评分机制保留。这里的 RAG Only 指删除 few-shot 参考示例，并不表示所有参考检索或数据访问都消失。

删除清单：[15 条 CSV 示例](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_examples.csv)；[79 条无 CSV 示例](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_query_only_examples.csv)。

### 6.3 Few-shot Only（当前 NRM 直接数值模型版本）

[Few-shot Only notebook](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb)

分类复用同一最终全量缓存；各 route 的建模与代码参考示例保留。移除本题 CSVQA 工具和 LLM 抽取计划/执行流程。Python 用 dtype=str、keep_default_na=False 读取完整原始 CSV，保留全部行列和解析后的文本值，直接提供给建模 LLM；没有另一个 LLM 在建模前筛选、摘要、重写或补造数据。

NRM/RA/TP/AP/FLP：完整 Python Observation → 数值 Formulation（包含所有必需 ID、系数、向量/矩阵）→ 原问题 + Formulation + 代码示例生成自包含 code → 求解。ReAct parser/executor 保留，但当前数据工具列表为空。当前 NRM 与其他四个数值 route 一样，不再传专用结构 schema，不再依赖或注入 CSVQA_DATA；RA 也不注入 LEGACY_RECORDS。

代码生成提示不直接追加完整原始 Observation；Formulation 中必要的具体系数仍可以进入代码 LLM，所以不能说该 LLM 完全看不到任何数值。执行程序从 Formulation 转写的程序数据获取系数。

Others：建模输入由概要改为完整 Python Observation；仍生成抽象计划；代码 LLM 只接收字段结构、计划、原问题、路径和代码示例，不直接接收完整原始数据。生成程序执行时读取完整 CSV。原 Others 没有 CSVQA，该差异是数据呈现方式，不能单独归因于移除 CSVQA。

NRM 直接数值 Formulation 是用户追加要求的局部改动，已独立完整重跑。旧 92/101 版本使用 NRM 符号模型 + 结构 schema + 执行时原始数据注入，旧结果独立保留在 /Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_full_review_v6_completed_r2。

### 6.4 LOTO Examples And Route

[LOTO notebook](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb)

按七个真实类别划分 fold；真实类别仅用于确定该 fold 要排除哪类样本，不提供给 agent 指定替代 route。每个 fold 从分类 RefData 和建模/代码示例库移除目标类别样本；清除分类与示例缓存，重建剩余参考检索库。分类 agent 必须在允许标签中重新分类，不复用全量预测。

将目标类别对应的执行 route 禁用；在分类结果、示例检索、建模及代码生成入口验证 route 允许。禁止 route 的提示/API 分支不会被调用。选择其余允许 route 后，使用该 route 剩余示例及原全量数据流程完成建模、代码和求解；没有根据参考答案挑 route 或替换失败代码。

Mixture 与 Others 对应同一个 Others 执行 route：其中任一 fold 禁用 Others 时，两种标签都不能被选作分类输出，但参考库样本排除仍按 held-out 真实类别执行。

| Fold | 移除示例数 | 剩余示例数 | 禁用 route |
|---|---:|---:|---|
| TP | 1 | 14 | TP |
| NRM | 1 | 14 | NRM |
| RA | 1 | 14 | RA |
| FLP | 1 | 14 | FLP |
| AP | 1 | 14 | AP |
| Mixture | 8 | 7 | Others |
| Others | 2 | 13 | Others |

101 条记录中分配至禁用 route 的数量为 0。[每个 fold 的实际替代 route 分布](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/current_complete_results/loto_folds_and_routes.json)。

LOTO 的通用优化仍使用本题语义、合法原始数据、标识符/矩阵映射和共同建模规则；不会重新加入被移除类别的示例，不读取测试参考模型或参考答案辅助生成。参考答案仅在事后评价使用。

## 7. 全部失败与不匹配

全部有效记录 755 条：全量 452 + 三项主集各 101。Objective 不匹配/失败共 48 条（全量 7、RAG 16、Few-shot 13、LOTO 12）。每一条均保存在 unsuccessful_cases.csv，含来源 CSV、真实类别、实际 route、目标值与参考值、错误类型、模型与代码路径。


## 后续：606 强制 route 实验

101 主集 × 六个执行 route 已完成：复用 101 个同 route 全量结果，新跑 505 个组合。共 527/606 Objective match、582/606 Solved。每个 route 分别为 TP 82、NRM 90、RA 90、FLP 85、AP 87、Others 93 /101 Objective match。该矩阵与上述全量主集有 101 条复用重叠，单独保存，不将其视为新增 606 个独立题目。

[完整 606 route 报告](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/report/report.md) · [逐题匹配矩阵](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/report/objective_match_matrix.csv)
