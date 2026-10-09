# 最终通用全量版本：各 route 的实际流程

对应交付代码：[LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb)；冻结评测源码为 [full_review_v6.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/candidates/full_review_v6.ipynb)，SHA256：`be8e63e5a205d0c524c02cbc04d9ae098bf881e4e302e65f119e52079d094c33`。交付版仅文件名、输出目录、说明和 provenance 不同，推理函数及提示常量一致。全量覆盖结果为 445/452 Objective match、450/452 Solved，已通过门槛和来源核对。三份组件实验已从该版本派生，并完成各 101 题的新 r2 轮次：RAG Only 85/101、Few-shot Only 的旧 r2 为 92/101、LOTO Examples And Route 89/101（Objective match）；Solved 分别为 91/101、96/101、98/101。Few-shot Only 当前 NRM 直接数值模型版本已完整重跑：88/101 Objective match、94/101 Solved。新轮次均无额度错误。RAG Only 前一轮因 API 耗尽额度而不能用于性能比较，原始记录独立保留。

## 共同入口与评分

1. 输入本题的自然语言问题和 CSV 路径。
2. 分类 ReAct agent 调用一次 FileQA，检索参考库中最相似的五条 `prompt + Type`，返回七个语义类别之一：TP、NRM、RA、FLP、AP、Mixture、Others。分类过程不读取测试题的真实类别或参考目标值。
3. TP、NRM、RA、FLP、AP 使用各自 route；Mixture 和 Others 都进入 Others。没有外部 CSV 的题目也进入 Others 的无 CSV 分支。
4. 建模和代码示例是允许使用的训练参考样本，示例系数、维度和标识符不能直接复制到当前题。
5. 生成代码由共同执行器运行，读取真实 Gurobi 模型对象 `m`（兼容 `model`）。只有求解器状态为 OPTIMAL 才计入 Solved；随后以 `rel_tol=abs_tol=1e-4` 匹配参考目标值，计入 Objective match。参考答案在生成结束后评分，不传入生成提示。

## 六个执行 route

| Route | 本题数据进入建模的方式 | 建模阶段 | 代码生成与数据来源 |
|---|---|---|---|
| **NRM：网络收益管理** | **planned**：CSVQA 让 LLM 根据问题和完整表概要提出 JSON 抽取计划；Python 校验并执行表、列、过滤与矩阵映射 | ReAct 调用一次 CSVQA；检索 1 个建模示例；输出符号模型和精确 Data Mapping，不逐行抄数值 | 检索至多 2 个代码示例；将模型、原问题和已验证的完整抽取结果交给代码生成；运行时注入 `CSVQA_DATA`，代码不得另读 CSV 或猜数值 |
| **RA：资源分配** | **legacy CSVQA**：源 CSV 全部行作为上下文，LLM 按本题要求返回来源行，Python 检查来源、字段、值及重复行数量 | ReAct 调用一次 CSVQA；检索最多 3 个建模示例（当前 RA 库中只有 1 条）；生成含标识符和系数的数值模型，明确全局活动与按资源分配的区别 | 检索至多 2 个代码示例；代码生成同时收到原始 Observation 的记录；运行时注入 `LEGACY_RECORDS`，从记录读取系数，不读外部 CSV |
| **TP：运输** | **legacy CSVQA**，同样执行来源行校验 | ReAct 至少调用一次 CSVQA；检索最多 3 个建模示例（当前 TP 库中只有 1 条）；输出完整数值模型，保留供给、需求、成本与轴标识符 | 检索至多 2 个代码示例；依据数值模型和原问题生成自包含代码，系数写入代码，不读取外部 CSV |
| **AP：指派** | **legacy CSVQA**，保留成本以及资格、可用性等源字段；同样执行来源行校验 | ReAct 至少调用一次 CSVQA；检索 1 个建模示例；按问题给定的资格等级顺序判断配对可行性，不能把“存在报价”等同于“允许指派” | 检索至多 2 个代码示例；依据数值模型和原问题生成自包含代码，实施资格与指派约束 |
| **FLP：设施选址** | **legacy CSVQA**，保留设施/客户 ID、固定成本、容量、需求及成本矩阵；同样执行来源行校验 | ReAct 至少调用一次 CSVQA；检索 1 个建模示例；生成设施启用与二维运输模型，保留矩阵方向与对应关系 | 检索至多 2 个代码示例；依据数值模型和原问题生成自包含代码，不自动补零、转置或添加商品维度 |
| **Others：其他/混合结构** | Python 读取源 CSV，提供字段、类型、前 10 行预览、完整表统计和与问题有关的匹配证据；概要不会替代完整数据 | 检索 3 个参考样本；延续原有 LLMChain 输出抽象建模计划 | 第二个 LLMChain 接收计划、概要、问题、文件路径及参考代码；生成的 Python 在执行时读取所需 CSV 的完整内容，并按问题筛选、连接和建模 |

### ReAct 的实际边界

分类，以及 NRM、RA、TP、AP、FLP 的建模使用 ReAct。**带 CSV 的 Others 继承原版“概要 → 抽象计划 → 代码”的两个 LLMChain，不是 CSVQA ReAct。** `CSVQA_MODE_BY_ROUTE['Others']='legacy'` 不表示该分支会调用 CSVQA。它没有本题数据抽取 planner，状态记录为 `LEGACY_SCHEMA`。

没有 CSV 的 Others：检索 `RAG_Example_Others_Without_CSV.csv`，由带 ORLM_QA 工具的 ReAct 生成数学模型，然后通过共同代码生成器和求解器执行。问题中的数值构成当前数据来源。

当前参考库共 15 条：AP 1、UFLP/FLP 1、NRM 1、RA 1、TP 1、Mixture 8、Others 2。Others route 的检索候选合并 Mixture 与 Others；各 route 的 k 是检索上限，不会补造不足的示例。RAG Only 删除这些样本在建模和代码生成中的使用，明细见 [删除示例表](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_examples.csv)。

### RAG Only 删除示例清单

下表行号为 `RAG_Examples_All.csv` 的零基索引。所有示例在建模与代码生成中排除；两份消融仍复用全量分类结果。

| 行号 | 类别 | 示例问题 |
|---|---|---|
| 0 | AP | 施工经理与项目指派 |
| 1 | UFLP/FLP | Superstore 分店补货与供应商启用 |
| 2 | NRM | Nike 鞋款销售与配额 |
| 3 | Mixture | 三种产品、22 天生产计划 |
| 4 | Mixture | 两种产品与钢材、铁材、机时资源 |
| 5 | Mixture | 三种化工产品的原料混配 |
| 6 | Mixture | 煤、电、钢三部门投入产出 |
| 7 | Mixture | 十种金融产品组合与风险限制 |
| 8 | Mixture | 七个行业的银行贷款分配 |
| 9 | Mixture | 四台设备启用与生产 |
| 10 | Mixture | 三种组件、四个车间的工时分配 |
| 11 | Others | 四个月的仓库租赁 |
| 12 | Others | 面粉与大米采购 |
| 13 | RA | 超市商品向配送中心分配 |
| 14 | TP | Walmart 供给、客户需求与运输成本 |

无 CSV 分支另排除 `RAG_Example_Others_Without_CSV.csv` 的 79 条参考行，删除三个固定示范 INTEGER、MULTI-PERIOD FLOW、LOGIC+BINARY，以及只检索这些参考示例的 ORLM_QA。这不影响用于当前题原始数据的 CSVQA。明细见 [无 CSV 示例清单](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/rag_removed_query_only_examples.csv)。本轮主集 101 题都有外部 CSV，未评测无 CSV 分支。

## 回退与通用规则

- **NRM**：计划格式、过滤证据或数据映射不合法时，保存原计划与错误，回退到 Python 读取的完整原始数据，仍由相同建模 agent 按原问题确定所需范围。当前关闭建模输出截断后的额外重试。
- **RA、TP、AP、FLP**：legacy 返回的来源行存在字段改写、数值改写、派生行、外来行或额外重复时，保存原响应与原因，回退到完整原始来源行。未把这些 route 改为 planned。
- 来源行校验保证已返回行的忠实性，**并不证明合法子集已覆盖全部所需行，也不证明筛选语义正确**；这些问题仍可能导致目标值不匹配。
- 各 route 按本题决定变量域、可选整数批量、双向启用关系、条件/无条件界、目标常数和单位。键、矩阵轴及分表映射须一致；不得补造数值。
- 模型快照 `gpt-4.1-2025-04-14`，`temperature=0`、`top_p=1`；embedding 为 `text-embedding-ada-002`。SDK 重试、求解机制及评分容差与基线保持一致。

## 在此全量版本上派生实验

| 实验 | 分类 | 建模、代码示例 | 本题数据流程 |
|---|---|---|---|
| **RAG Only** | 复用最终全量主集分类结果 | 删除各 route 的建模和代码示例；无 CSV 分支的示例及固定示例触发段也删除 | 保留 CSVQA 的 legacy 行上下文、来源校验和回退；保留 NRM planner/Python 抽取；保留 Others 概要与执行时完整读 CSV |
| **Few-shot Only** | 复用同一分类结果 | 保留各 route 的参考示例与共同数学要求 | Python 把完整原始 CSV 提供给建模阶段，无 LLM 数据预处理；删除 CSVQA 和本题抽取 planner；NRM/RA/TP/AP/FLP 输出完整数值 Formulation 后直接生成自包含代码，不注入运行时数据；完整原始 Observation 不直接进入代码生成；Others 仍由代码运行时读取 CSV |
| **LOTO Examples And Route** | 按目标类别划分 fold，移除该类别分类样本后，在允许的类别中重新分类 | 分类、建模、代码示例库均排除目标类别；使用重新选中的 route 的剩余参考样本 | 禁用目标类别对应的 route，在分类输出、示例检索、建模及代码生成前校验；数据流程由允许的所选 route 决定 |

Few-shot Only 的 NRM 数值 Formulation 流程为 2026-10-06 用户要求的后续调整，已完整重新运行：88/101 Objective match、94/101 Solved；此前 92/101 Objective match、96/101 Solved 对应调整前版本。

LOTO 的 Mixture 与 Others 都对应执行 route Others；禁用 Others 时，两种语义标签均不能作为分类输出。真实类别只决定 fold，不直接指定替代 route。

完整全量验证已达到约定门槛；派生版本已通过离线边界检查，并完成新的消融及 LOTO。派生差异限定为指定组件移除，共用数学提示、模型与执行参数一致。单次实验的性能差异同时受到生成波动影响，未隔离每一条具体改动的独立因果贡献。

相对最终全量主集的 97/101 Objective match，RAG Only、当前 Few-shot Only、LOTO 分别少正确 12、9、8 题。相对前一有效组件版本，RAG Only 从 83 增到 85，Few-shot Only 从 86 到旧 r2 的 92，最新 NRM 直接数值模型版本为 88，LOTO 从 91 降到 89；LOTO 的 Solved 从 97 增到 98。LOTO 的三项失败包括来源标识符不一致、API 单条输入长度超限及 API 请求超时，另外有九题已求解但目标值不匹配，均保留在分母。

## 恢复运行的范围

按用户本次要求，剩余六个冗余列 sheet（100pct-S1/S2/S3、200pct-S1/S2/S3）各 35 题，共 210 题，已使用相同冻结候选源码完整重跑，Objective match 与 Solved 均为 209/210。已有主集、Variants、50pct 三组整组保留；全量汇总已记录每组来源批次及源码、输入哈希，不按单题选择不同轮次的结果。旧额度失败记录独立保留。

这属于同一代码版本在两个 API 批次上的完整数据集覆盖；不是新的一轮连续 452 题运行。最终报告已明确记录批次来源。
