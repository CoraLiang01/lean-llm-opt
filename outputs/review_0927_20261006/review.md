# 0927 全量代码审阅记录

## 范围与比较规则

逐单元审阅当前全量、RAG Only、Few-shot Only、Examples And Route LOTO 四份 notebook 的共用流程、差异流程和实验入口，以及对应的构建、运行、评分、缓存与边界检查脚本。源码副本、函数清单、改动明细和校验记录均在本目录。

当前基线为完整新跑的 full_v1：Objective match 430/452，Solved 442/452；主集分别为 92/101、96/101。历史主集 95/101、99/101 另外标识，不能当成本轮同时运行的基线。

按用户最新要求，最终候选需在完整 452 题上的 Objective match 总数超过 430；允许个别数据集下降。原目标为 Variants 至少 32/36、九个冗余列 sheet 各至少 31/35；Solved 单独报告。门槛通过前不启动新的消融或 LOTO。只比较完整版本，不拼接不同版本的单题最优结果。

另发现 Examples Only 的同名副本配置指向另一组 ReAct 实验（多个 route planned、SDK retry=2），其参数不属于本轮四份文件的共同基线。本轮记录了这一差异，保留该文件及独立 ReAct 工作的修改。

## 已确认并局部修复的问题

| 位置 | 问题 | 最小修复与理由 |
| --- | --- | --- |
| CSV 读取 | 连续编号的 Unnamed/index 列被直接删除；空列也可能是矩阵的合法轴 | 保留全部解析后的行、列、文本值；仅处理已有 BOM 规范化 |
| legacy Observation 解析 | 对列名 strip 会让 A 与 A空格覆盖彼此；重复表头可能丢数值 | JSON 保留原始列键；CSV/Markdown 重复列明确报错 |
| planned 条件执行 | 数值列 prefix/contains 会抛出未归类的 AttributeError；多元素 eq 数组可能按行误比较 | 类型、值形状验证；单元素数组按一个标量处理，多元素保持拒绝 |
| planned 表清单 | 非对象表项在验证前调用 get；物理文件可有多个视图，错误信息却称只能使用一次 | 先验证对象和索引类型；保持多个视图，明确覆盖规则 |
| planned 证据 | 原来的去标点规则把 -1 与 1 混为同一证据 | 保留正负号和标点，只兼容空白、大小写与引号排印差异 |
| planned 矩阵轴 | 去标点会合并 A-B/AB；合法的不同轴顺序会被拒绝 | 保留标点和大小写，按键集合验证覆盖，并记录源顺序是否一致，不重排数据 |
| planned 分类片段 | 匹配计数没有显示实际标签；抽象模型可能再次按猜测的完整标签筛掉数据 | 增加实际匹配标签及通用 exact/prefix/contains 选择要求；有效过滤后的数据直接使用 |
| Others 代码提示 | 模型名 m/model 与最终 m 接口、顶层求解与函数求解要求相冲突 | 明确顶层 m；保留原执行器的兼容读取机制 |
| 共用代码提示 | 禁止外部 CSV 的流程又收到强制读取 CSV 的要求 | 把读取要求限定在允许读取的任务中 |
| pandas 生成代码 | Series.casefold、itertuples 任意列名访问、Series.to_dict 非业务 ID 索引会导致运行错误 | 使用合法字符串接口、显式列访问、显式 ID/value 字典；矩阵列与实体键保持一致 |
| 多表键 | 将 opaque ref、entity_id 与显示名称直接当作同一种键 | 保留一致的 opaque 键，仅在跨命名空间时用源 lookup 对齐并校验覆盖；没有映射时不得猜测 |
| 可选批量与启用关系 | 授权被写成强制选取，类别启用标志缺少成员到类别的约束，可能漏计费用 | 根据 query 保留零选择，使用双向逻辑关联，不依赖目标系数的正负来代替约束 |
| 分表与版本处理 | 按猜测的文件位置取表，遗漏分表、部分表未执行版本/删除规则 | 按字段和 table tag 收集全部对应行；先按 query 筛选有效记录，再转数值、join、sum |
| AP 资格 | 列出的成本被当作可行配对，等级比较可能误用字典顺序 | 明确保留资格/可用字段，按 query 给出的等级排序验证每个配对 |
| ReAct 输出格式 | 建模模型有时直接输出正文而缺少 Final Answer 标记，默认 parser 拒绝 | 在共用前缀明确结束标记；重复章节与重复数据只输出一次，不改变 parser 或执行次数 |
| AP 来源结构 | CSVQA 将实体/资格/报价表拼成派生行，可能错写资格或漏报价 | 保持原表、列名、行值，后续建模阶段实施题目资格判断 |
| 生成 API | Index 被调用字典 get；Gurobi TempConstr 被传给 quicksum | 提示合法列检查、约束接口与二元 prerequisite 链接，保持执行器错误传播 |
| legacy 来源忠实性 | LLM 可返回拼接的派生字段、错误数值或多余重复行，提示要求没有实际校验 | Python 按原 source/字段/值/行重复数校验；不符合时记录原响应与原因并回退到完整原始证据，保留 legacy/ReAct，由模型按原 query 选取 |
| 文件位置 | 参考库和示例备用数据路径隐含依赖工作目录 | 按 PROJECT_ROOT 解析，支持配置根目录后从其他工作目录运行 |
| 仅问题流程 | 记录/建模使用 Others，而代码提示仍使用分类得到的另一 route | 三个阶段统一原有的 query-only Others 路径 |
| 失败续跑 | 同一结果目录可能重跑失败题并覆盖旧记录；早期失败缺少部分文件，不能恢复 | 恢复完整或失败记录；拒绝改代码/数据后覆盖旧轮次，损坏记录明确报错 |
| 布尔结果 | 字符串 False 在常规布尔判断中为真 | 对超时及 route_allowed 等字段显式解析 |
| 运行中断 | KeyboardInterrupt/外部截止时间可能丢失已完成阶段的信息 | 保留阶段上下文后继续抛出中断，由原评测器记录失败 |
| 数据集清单 | 重复/空题号或缺少评分值可能造成覆盖或错误分母 | 题号在调用前校验；本轮 452 题全部有数据文件和有限参考目标值 |
| 消融派生 | Few-shot Only 可能漏掉全量共用的无条件界提示；当前任务提示仍要求已移除的 CSVQA | 最终派生时保留共用数学要求，只移除当前工具要求，参考示例内容保持 |
| 分类缓存 | 需明确对应最终全量结果，而不能误用旧全量缓存 | 最终全量通过后生成只含预测的缓存；校验源码、缓存哈希和题目/数据对应关系 |
| RAG 文档 | 旧审计称源 CSV 有向量检索，实际 CSVQA legacy 使用完整行文档的 stuff chain | 按实现描述：源 CSV 上下文/抽取保留；FAISS 用于参考示例，不虚称源数据向量检索 |

## 校验和保留条件

- notebook 格式、每个代码单元编译、按顺序加载定义、全部数据文件及评分值检查。
- 合成数据与模拟调用复现解析、过滤、键对齐、目录变化、续跑和中断问题；不访问模型 API。
- 模型 gpt-4.1-2025-04-14、temperature=0、top_p=1、embedding、SDK retry=1、匹配容差 1e-4 保持一致。
- NRM 延续已有 planned；其他 route 延续 legacy；保留原有 ReAct parser/executor 与各阶段职责。
- 推理只接收题目、当前数据与允许的训练参考示例。参考目标值仅作为记录元数据与事后评分使用，不传入生成提示；LOTO 真实类别只用于 fold，重新分类并在建模/代码生成前拒绝禁用 route。
- 没有题号、预设目标值或针对单个 sheet 的生成分支。新的数据集/问题仍需实测，现有基准不能保证任意未来任务的准确率。

## 运行记录

每版各自冻结源码，并保存逐题 prompts、用量、运行日志、原始 result.json、模型、Observation、求解代码、解与错误。修改前源码及所有中间完整评测均保留。此文件中的离线修复不等同于已通过性能门槛。

API 返回 credit_balance_exhausted 后，用户要求“先保留代码和当前记录”。运行已停止，最新候选 full_review_v6 另存于 candidates，根目录四份基线未被本轮新候选覆盖。候选通过 18 项离线检查，完整主集 Objective match 为 97/101、Variants 为 35/36、前三个冗余列 sheet 为 34/35、35/35、35/35；其余六个 sheet 的有效验证尚未完成，整体性能门槛尚未确认。所有实际额度失败与中断原样保留，新的消融及 LOTO 未启动。结果与逐题记录见 [审阅汇总报告](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/review_0927_20261006/report/report.md)，恢复规则见 paused_control.json。

最新 gate 规则见 gate_policy.json 和各版 gate_full_review_vN.json；baseline_manifest.json 的 gate 字段保留初始规则作为历史记录，运行门槛由最新规则决定。

## API 恢复后的续跑

用户随后确认“已更新密钥，可以继续”。既定 chat 模型和 embedding 调用检查通过；按本次请求，六个剩余冗余列 sheet 各 35 题整组重跑。推理源码没有继续修改；运行脚本仅增加整组选择参数。已完成五组与续跑六组按预先固定的来源汇总，并明确记录两个 API 批次；旧额度失败记录不覆盖。汇总工具经 452 条合成记录离线检查，确认能发现全部原始结果、核对来源哈希，并拒绝额度失败污染的组。新的消融/LOTO 仍需通过全量整体门槛后启动。详情见 resume_plan.json、resume_consistency_check.json、resume_consolidator_checks.json 和 route_guide.md。

全量来源核对及门槛通过：445/452 Objective match、450/452 Solved。四份根目录 `_0927.ipynb` 已更新到经离线组件边界检查的派生版本，原始四份仍在 baseline/。本次恢复未修改推理源码；交付全量仅修改文件名、输出目录、说明和 provenance。RAG Only、Few-shot Only、LOTO Examples And Route 将依次完成各 101 题。

组件实验 RAG Only 第一轮产生 61 条 API credit_balance_exhausted，另有真实生成截断/代码错误；该轮原始计数不作为消融性能结论。用户再次更新 API，既定模型和 embedding 检查通过，开启 r2 的完整 101 题；源码和分类缓存不变。运行器新增统一的外部额度中断处理，取消未执行的排队任务，保留真实在途结果；不改变每题推理和 SDK retry。

## 本次恢复的最终状态

新 r2 轮次三项实验均已完成各 101 题，额度错误为零。RAG Only 的 Objective match / Solved 为 85/101 / 91/101；Few-shot Only 为 92/101 / 96/101；LOTO Examples And Route 为 89/101 / 98/101。相对最终全量主集 97/101 / 101/101，Objective 分别少 12、5、8 题。

LOTO 相对前一有效组件轮次 91/101 少正确 2 题，Solved 从 97/101 增到 98/101，不能宣称本轮 LOTO 准确率提高。LOTO 三个失败为来源键不一致、API 单条输入长度超限、API 请求超时，另有九个目标值不匹配。全部失败、超时与不匹配均计入分母；未挑选单题结果或按参考答案修改生成。

四份交付 notebook 哈希与冻结交付清单一致，格式与所有代码单元编译检查通过。最终静态及实际运行边界审计通过：两份消融分类缓存与全量一致；Few-shot Only 的 101 份完整原始 Observation 共 289332 个单元格经源 CSV 比对，完整 Observation 未直接进入代码生成；LOTO 所有 101 题的禁用 route 使用次数为零。全量 445/452 Objective match、450/452 Solved，由同一冻结源码在两个预先固定整组来源的 API 批次覆盖完成。详见 final_boundary_audit.json、final_delivery_format_check.json、route_guide.md 和 report/report.md。
