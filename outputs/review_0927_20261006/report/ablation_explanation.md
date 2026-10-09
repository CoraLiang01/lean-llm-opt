# 两项消融实现、分数解释与全量改动

本说明对应已完成的 r2 运行版本及其逐题结果，旧分数保留。

**2026-10-06 后续代码调整：** 按用户要求，当前 Few-shot Only notebook 的 NRM 已改为“完整 Python Observation → 数值 Formulation → 自包含代码 → 求解”，不再通过结构 schema 生成依赖 `CSVQA_DATA` 的代码，也不再注入运行时数据。下面关于 NRM 注入及 Few-shot Only 92/101 的说明属于调整前的已运行版本；新版本已完整重新运行，Objective match 为 88/101，Solved 为 94/101；不沿用旧评分。改动及原版备份保存在 `outputs/review_0927_20261006/nrm_direct_formulation_revision/`。

## 实际实现

### RAG Only

- 分类：复用最终全量的预测分类缓存，不读取真实类别指定 route。
- 建模和代码示例：`retrieve_rag_examples` 返回空列表，因此 `build_formulation_examples` 和 `retrieve_csv_code_example` 不再加入训练参考问题、数学模型或代码。CSV 参考库的 15 行全部排除；无 CSV 分支的 79 行参考及三个固定示范也排除。
- 数据：保留 CSVQA。NRM 保留 planned JSON 计划与 Python 校验执行；RA/TP/AP/FLP 保留 legacy 完整来源行上下文、LLM 按题抽取及来源校验；Others 保留概要、抽象计划和执行时完整读取 CSV。
- 仍保留：共同数学规则、route 职责说明、代码规范、Gurobi 执行和相同评分容差。它不是完全没有任务说明的 zero-shot 对照。
- 当前主集实测 85/101 Objective match、91/101 Solved；相对全量 97/101 少正确 12 题（11.88 个百分点）。真实类别 NRM 的下降最大：24/25 → 17/25。

### Few-shot Only

- 分类：复用同一最终全量预测缓存；建模、代码参考示例均保留。
- 数据入口：`direct_source_payload` 用 Python、`dtype=str`、`keep_default_na=False` 读取本题全部 CSV，保留所有解析后的行列和文本值，作为完整 Observation。没有 LLM 在建模前筛选、摘要、重写或补值。
- 工具：移除当前数据 CSVQA 与抽取 planner/executor。NRM/RA/TP/AP/FLP 继续采用原 ReAct parser/executor，但当前工具列表为空。
- 代码生成：完整原始 Observation 不直接追加进提示。提示接收建模输出、原问题、参考代码，以及适用时的结构信息（来源、表 ID、字段、行数等，不含原始单元格 records）。建模输出中必要的数学系数仍可进入代码生成。
- 执行：Few-shot Only 的 NRM 在代码生成完成后，由 Python 将完整原始数据写入运行变量 `CSVQA_DATA`；Others 的生成代码在运行时读取完整 CSV；RA/TP/AP/FLP 按数值模型生成自包含代码，当前 Few-shot Only 不向 RA 注入 `LEGACY_RECORDS`。完整数据不直接进入代码生成 LLM，不等于执行程序没有原始数据。全量/RAG Only 的 RA 使用 `LEGACY_RECORDS`，不能将该机制混同于 Few-shot Only。
- Others 的特别边界：全量该分支本来使用两个 LLMChain，没有 CSVQA。为了执行“Python 全量 Observation”的既定要求，Few-shot Only 把其建模输入从概要改成完整数据，代码阶段只给结构信息。因此这个分支的差异是数据呈现方式，不能单独解释为 CSVQA 工具的贡献。
- 运行审计验证 101 份完整 Observation，共 289332 个源单元格；数据未被 LLM 预处理，完整 Observation 未直接传到代码生成。
- 当前主集实测 92/101 Objective match、96/101 Solved；相对全量少正确 5 题（4.95 个百分点）。TP/RA/FLP/AP 共 50 题全部正确，下降集中在 Mixture（三题）、Others（一题）、NRM（一题）。

## 为什么仍能达到较高准确率

代码和逐类结果支持以下解释，但单轮评分不能隔离每个因素的独立因果贡献：

1. 两份 Only 仍保留全量分类路由、GPT-4.1、通用建模和代码提示，以及 Gurobi。它们移除的是指定组件。
2. 新增通用提示包含可选整数批量、双向启用逻辑、资格等级、条件/无条件界、业务键和分表处理等规则。这些指导也随全量同步到两份消融，降低了对单个参考示例的依赖。
3. Few-shot Only 的完整原始数据是较强输入，同时避免了额外 LLM 抽取产生的漏行或改值风险；数据工具移除不保证准确率必然降低。
4. 当前 Few-shot Only 在四个标准类别上达到 50/50；其余类别仍有缺口。RAG Only 已有十二题的总差距，其中七题来自 NRM。

逐题配对也说明差异不是所有题目单调下降：RAG Only 相对新全量新增 13 个错误、改善 1 题（OR-087），净少正确 12 题；Few-shot Only 新增 7 个错误、改善 2 题（OR-080、OR-016），净少正确 5 题。改用完整原始数据与模型生成波动都可能影响这种变化，当前单轮结果不能拆分二者的独立效应。

在保持既定定义时，应依据组件边界修复残留、按完整数据集报告，而不是按目标分数调节其他条件。若要研究额外指导的贡献，可单独增加“无示例且无通用建模规则”消融，明确其同时移除两个组件。既有 Variants 与九个冗余列 sheet 也适合用同一全量与消融配对评估数据规模和复杂度；准确率方向应由实际完整结果决定。

## 相对原全量的具体改动

本轮审阅前的完整比较基线是 full_v1（101 题 92 Objective / 96 Solved），当前是 97 / 101；更早的 final_101_V2 历史全量为 95 / 99。历史分数独立保留，没有把不同轮次单题最优结果拼接。

| 范围 | 当前实际改动 | 解决的问题 |
|---|---|---|
| CSV 读取与字段解析 | 保留 index/Unnamed、空列和原始列键；拒绝歧义重复表头 | 避免合法业务键、矩阵轴和数值被删除或覆盖 |
| NRM 计划校验 | 字符串运算类型、单元素标量、证据标点/正负号、文件覆盖、矩阵键与顺序检查；显示实际匹配类别值 | 减少错误筛选、标识符碰撞和合法轴顺序被误拒绝 |
| legacy 来源校验 | 对来源、字段、值和重复次数逐项验证；非法结果保存原因并回到完整原始证据 | 防止派生行、改值、外来行或额外重复进入模型。最终全量有 9 题使用此回退，全部在扩展集；主集未触发，不能把主集提升归因于此回退 |
| 数学规则提示 | 可选选择、双向启用、资格等级、完整分表、版本/删除筛选、业务键与单位一致 | 减少漏约束、错误资格与错误数据聚合 |
| 代码生成提示 | 正确使用 Series/Index、列名、ID 字典、Gurobi TempConstr；统一顶层存活模型 m；读取要求按当前数据机制限定 | 修复 pandas/Gurobi API 错误与相互冲突的接口要求 |
| 记录和续跑 | 失败也保留；源码/数据变更使用新目录；正确解析 False；保存抽取 trace 和中断上下文 | 避免覆盖失败或产生错误指标；不应当算作建模能力提升 |

已有实际例子：full_v1 的 OR-082（Series.casefold）、OR-085（itertuples/Unnamed 列）、OR-027（类别片段被当作完整标签）失败，本轮均正确。主集按真实类别净变化为 RA +1、Mixture +2、Others +2，其余 Objective 总数不变；个别题仍有退化，详见逐题记录。

框架、模型快照、采样参数、NRM planned/其他 route legacy、求解执行器和匹配容差沿用原配置。若比较更早的 full_before，另有一项早期调整：SDK 网络重试从 0 改为 1，并记录重试；当前全量和所有组件实验统一使用 1 次 SDK 重试。

- [本轮审阅前基线与当前的逐行差异](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/full_vs_baseline.diff)
- [更早 full_before 与当前的逐行差异](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/full_vs_earlier_before.diff)
- [101 题按真实类别的完整统计，含历史全量](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/main_101_per_class.md)
- [可用于进一步分析的类别结果 CSV](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/main_101_per_class.csv)
