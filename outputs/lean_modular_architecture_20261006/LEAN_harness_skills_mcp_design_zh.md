# LEAN 的 Harness、Skills、RAG 与 MCP 模块化方案

本方案以 `LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb` 为准，是架构设计和代码组织建议，不是已经实现、运行验证的迁移版本。没有修改原 notebook，也没有修改 AE 回复文档。

## 1. 建议定位

建议描述为：**由运行框架（harness）协调类别建模技能（skills）、参考案例检索和实例数据工具的优化建模系统。**

这是工程组织方案，不是所有系统必须遵循的一项统一标准。Skills、MCP、RAG 分别涉及程序指导的表示、工具与数据的接入、检索结果如何参与生成。采用其中一种不要求同时采用全部。

- Harness：管理模型调用、阶段顺序、状态、上下文、工具执行、错误和结果。
- Skill：描述可复用的类别建模方法，包含适用条件、建模指令、数据使用规则和输出约定。
- Knowledge：专家参考问题、数据、数学模型和代码等内容。参考案例也可以包含程序性经验。
- Retrieval tool：搜索相关案例并返回案例内容。
- RAG：检索结果参与分类、建模或代码生成的完整机制；不是检索函数的另一名称。
- Data tool：访问当前实例数据，执行提取、筛选和验证。
- MCP server：将工具按 MCP 接口提供给客户端；它不替代具体检索、数据处理和求解算法。

## 2. 当前 notebook 的真实行为

| 环节 | 当前实现 | 模块化后的归属 |
|---|---|---|
| 分类 | `invoke_classifier`：ReAct 调用 FileQA 恰好一次；检索五个带类型标签的参考问题 | 分类 agent + 分类 guidance + reference tool |
| 类别路由 | `class_to_workflow_route`；Mixture/Others 共用 Others；无 CSV 时使用 Others | Harness 的路由策略 |
| 参考库 | `RAG_Examples_All.csv` 的 prompt、Data_address、Related、Required Data、Label、Code、Type | 共享 knowledge store |
| 类别内案例搜索 | `_get_rag_store`、`retrieve_rag_examples`：每类 FAISS，嵌入 prompt | Reference retrieval tool |
| 演示组装 | `build_formulation_examples`：把案例组装成 Question/Action/Observation/Final Answer | Workflow builder / demonstration renderer |
| 类别建模指导 | `get_NRM_response` 等函数中的类别指令与共享规则 | Skills 的 stage-specific instructions |
| NRM 数据提取 | LLM 生成计划；Python `_execute_plan` 执行；异常回退完整数据 | 复合 data tool 的 planned backend |
| RA/TP/AP/FLP 数据提取 | 全部源行送入 LLM；输出相关原始行；验证源行一致性；失败回退全部源行 | 复合 data tool 的 legacy backend |
| Others（有 CSV） | schema preview → abstract plan → 代码读取完整文件 | General skill + schema tool + 两阶段模型调用 |
| 代码生成 | `get_csv_code`、`get_code`、`_generate_code` | Code-generation stage，由 harness 调用 agent/model |
| 求解 | `_source_candidate`、`execute_code` | Execution/solver backend |
| 流程控制 | `execute_pipeline_case` 及模型调用循环 | Harness |
| 评估、缓存、结果文件 | benchmark loaders、finalize_record、缓存与物化结果 | Evaluation 与 artifact management |

**不能把当前全部 CSVQA 称为向量 RAG。** 类别参考案例采用向量检索；legacy CSVQA 把全部源行放入上下文，没有在目标 CSV 上做 top-k 向量检索。NRM 是计划驱动的数据提取。Others 的数据预览也不是完整数据提取结果。

## 3. 建议文件结构

以下名称是建议，不是已有模块。

```text
leanopt/
  runtime/
    harness.py                 # 流程与阶段状态
    model_client.py            # GPT-4.1 配置、超时、使用记录
    tool_dispatcher.py         # 本地/MCP 后端选择
    skill_loader.py            # SKILL.md 与阶段资源读取
    contracts.py               # 数据、模型、代码和求解结果约定
  agents/
    classifier.py
    formulator.py
    code_generator.py
  workflows/
    demonstrations.py          # 参考案例 → 演示
    routing.py                 # 七个标签 → 六条执行路由
    policies.py                # k、数据模式、调用次数等执行配置
  skills/
    classify-optimization/SKILL.md
    nrm-modeling/
      SKILL.md
      references/
        formulation.md
        data-extraction.md
        code-generation.md
    resource-allocation/SKILL.md
    transportation/SKILL.md
    assignment/SKILL.md
    facility-location/SKILL.md
    general-optimization/SKILL.md
    shared/
      query-semantics.md
      source-data-contract.md
  tools/
    reference_search.py
    csvqa.py                   # 保留现有对 agent 的复合接口
    data_profile.py
    planned_extraction.py
    legacy_extraction.py
    solver.py
  mcp/
    server.py                  # 暴露同一套工具实现
  evaluation/
    benchmark.py
    scoring.py
    cache.py
knowledge/
  reference_cases/              # 可先直接使用现有 CSV，不必转换格式
```

Skill 可以引用共享案例库和检索工具，不必将整个案例库复制到每个 skill。`scripts/` 是可选的；由 MCP 提供执行能力时，skill 可以主要由指令和参考说明组成。

## 4. 各条 route 的原有协议应保留

| Route | 建模阶段案例数 | 数据机制 | CSVQA 调用协议 | 主要输出 |
|---|---:|---|---|---|
| NRM | 1 | planned | 恰好一次 | 抽象符号模型 + 精确 Data Mapping |
| RA | 3 | legacy | 恰好一次 | 完整数值模型 |
| TP | 3 | legacy | 至少一次 | 完整数值模型 |
| AP | 1 | legacy | 至少一次 | 模型及完整必要参数；资格字段保留 |
| FLP | 1 | legacy | 至少一次 | 模型及完整必要参数；保留矩阵方向与 ID |
| Others，有 CSV | 3 | schema preview + 运行时全文件读取 | 不经过上述 ReAct CSVQA 协议 | 抽象计划 + Gurobi 代码 |

分类 FileQA 保留 k=5 与恰好一次调用。标准路由的代码参考检索保留 k=2，并保留当前对空 Code 的处理。Others 使用自身两阶段逻辑。无 CSV 的路径另行保留。

NRM/RA 对一次 CSVQA 的要求不能扩展成“所有路由恰好一次”。工具后端也不能擅自全部切换为 planned。

## 5. 工具接口建议

### 5.1 参考检索

建议两个工具，避免混淆不同的索引内容和用途：

```python
search_classification_examples(query: str) -> ClassificationExamples
search_reference_cases(route: str, query: str, k: int) -> ReferenceCases
```

第一项保留现有分类索引与五个例题；第二项保留按类别建立的 FAISS 索引。返回源案例各字段、可追溯标识和顺序，暂不额外增加 LLM 总结。

**检索工具返回证据，建模 agent 使用这些证据生成当前模型，两者共同实现参考案例驱动的 RAG。**

### 5.2 当前实例数据

第一阶段保留对建模 agent 的 `CSVQA(query)` 接口。工具的 route、文件和原始问题由 harness 绑定，避免 agent 随意改写执行路由或数据源。

```python
# 由 harness 创建绑定了当前任务的工具。
csvqa = make_csvqa_tool(
    route=route,
    dataset_address=dataset_address,
    original_query=query,
)
# 建模 agent 调用，后端保持 notebook 的 planned / legacy 行为。
evidence = csvqa(tool_query)
```

NRM 后端仍然执行：schema/profile → LLM extraction plan → Python execution → validation/fallback。保留原始值、源行顺序、过滤条件、表间关系、状态和 payload hash。

legacy 后端仍然执行：全部源行 → LLM 提取 → 源行一致性校验 → 必要时完整源行回退。当前校验主要保证输出行来自源数据且未篡改，并不能单独证明相关行已全部返回。

将来可以拆为 `profile_instance_data` 和 `execute_extraction_plan`。但让 agent 直接调用这两个工具，会改变工具调用序列与上下文，应作为后续方法变更评估，不能当作第一次纯重构。

### 5.3 求解

求解后端接收生成的代码，执行并返回状态、目标值、变量值和错误。若跨客户端共享，可再暴露为 MCP tool。当前本地调用也完全合理。

提取结果、执行结果尽量结构化；如果首次迁移改变了工具 observation 的格式，应明确记录这一变化，不能宣称提示完全等价。

## 6. NRM Skill 应怎样写

以下是设计示例，不是已安装的 skill，也不是已验证与旧 prompt 逐字等价的版本。

```markdown
---
name: nrm-modeling
description: Formulate network revenue management problems with external instance data after LEAN routes the problem to NRM.
---

# Network revenue management modeling

## Procedure
1. Use the NRM reference demonstration supplied by the workflow builder.
   Learn its modeling structure; never copy its instance coefficients.
2. Call the bound CSVQA tool exactly once before producing the formulation.
3. Use the validated returned data and preserve its original identifiers.
4. Construct a symbolic model with sets, parameters, variable domains,
   the complete objective, and every query-required constraint.
5. Provide Data Mapping using exact table identifiers and column names.
   Preserve validated selection predicates. If the tool returns full-data
   fallback, specify the query-supported selection explicitly.
6. Pass the formulation and data evidence to the code-generation stage.

## Supporting instructions
- references/formulation.md
- references/data-extraction.md
- references/code-generation.md

## Authority
The current problem description determines objective sense, variable
bounds, domains and additional requirements. Reference cases are examples.
Never invent missing coefficients or infer unstated replenishment.
```

技能复用的是上述方法与资源使用规则。相关案例、提取计划、目标数据和最终模型均可以随实例变化。生成的工作流是 skill 在当前任务中的执行；不需要固定全部具体动作和输出。

## 7. Harness 的伪代码

```python
# 架构伪代码：类和方法名称是建议，不能直接作为已完成实现运行。
def run_lean(request, runtime):
    classification = runtime.classifier.run(request.query)
    route = runtime.routing.select(
        classification.label, has_csv=bool(request.dataset_address)
    )
    skill = runtime.skills.load(route)

    # 保留每条 route 的原有案例数量与检索配置。
    references = runtime.reference_backend.search_for_route(
        route, request.query, runtime.policy[route]
    )
    context = runtime.workflow_builder.render(
        skill=skill, references=references, request=request
    )

    # 内部保留标准路由的 ReAct、Others 的两阶段和无 CSV 分支。
    result = runtime.route_runner.run(
        route=route, context=context, request=request
    )
    code = result.code or runtime.codegen.run(
        query=request.query,
        route=route,
        formulation=result.formulation,
        data_evidence=result.data_evidence,
    )
    runnable = runtime.execution.prepare(code, result)
    solution = runtime.execution.solve(runnable)
    return runtime.artifacts.save(request, result, runnable, solution)
```

工具调用循环可先由当前 LangChain ReAct 提供；外层 harness 负责统一管理。不能简单把 `execute_pipeline_case` 改名，就声称已完整迁移。

首次重构保留 `gpt-4.1-2025-04-14`、当前 embedding、采样配置、重试、检索和求解行为。自建 skill loader 读取 SKILL.md 并按阶段提供指导，与采用 OpenAI 托管技能执行 API 是两种接入方案；不能因为存在 SKILL.md 就假定 GPT-4.1 自动支持所有新 API 功能。

## 8. MCP 如何接入

```text
NRM skill：告诉 agent 如何利用案例和数据完成建模
    ↓
Harness：加载指导、组织上下文、执行模型与工具循环
    ↓
Tool dispatcher：选择本地调用或 MCP 调用
    ↓
Reference/data tool 的同一套业务函数
    ↓
FAISS 案例库 / 当前 CSV 文件
```

建议 MCP server 先暴露参考检索与 CSVQA。把求解作为独立后端，确实需要其他客户端访问时再暴露。

MCP 不必增加一个“专门的 RAG agent”。MCP reference-search tool 负责检索；返回内容进入模型上下文后参与生成。整个链路具有 RAG 功能。

远程 MCP server 不能凭本机路径字符串读取 Cora 的 CSV。需部署在能够访问数据的环境，或使用任务数据 ID 与明确的上传/挂载映射。首次本地迁移无需增加这种数据搬运。

## 9. 建议实施顺序与验证

1. **模块拆分。** 把 notebook 中的函数、原有 prompt 和 route 配置拆出；notebook 保留实验入口。对比渲染 prompt、检索结果、调用协议和数据 payload，优先保持既有行为。
2. **技能表示。** 建立 SKILL.md 与阶段参考文件，首先沿用原有指令，明确 harness 的加载与上下文注入逻辑。不要同时改写建模策略。
3. **MCP 适配。** 对已有工具添加 MCP 接口；本地和 MCP 使用同一底层实现。验证结构化结果、文件定位和异常语义。
4. **行为升级。** 如改用新的模型/API、原生 function calling、分拆数据工具、重排阶段或自动修复，分别作为后续改变评估。

验证须包括分类标签与路由、参考案例顺序、各 route 的调用次数、NRM plan/payload 与 fallback、legacy 源行保真、Others 完整文件读取、代码可执行性及目标值。模型生成有随机性，不能只凭一次目标值一致就证明重构完全等价。

评估真值、参考目标值和正确性判断留在 evaluation。Agent 请求和工具只能得到当前问题及其允许的数据，不应接收测试答案。

这种改写主要改善复用、维护、可追溯性和接入能力。性能提升必须通过新实验验证，不能由 Skills 或 MCP 的封装名称推断。

## 10. 对论文的影响

在完成迁移前，论文应描述当前“类别指导、参考案例与工具驱动的工作流”，可以说明它们与 skills/harness 的功能联系。

迁移并验证后，可描述工程实现为：

> LEAN-LLM-OPT uses a harness to coordinate category-specific modeling skills, reference-case retrieval, and instance-data tools. The skills encode reusable modeling procedures; retrieved cases provide supporting modeling knowledge; and data tools supply validated evidence from the current instance. MCP provides an optional interface for exposing these tools across agent clients.

如果这是修稿期才完成的代码组织变更，应说明与原实验实现的关系，不将新接口架构直接归因于旧结果。

## 官方依据

- OpenAI Skills：SKILL.md、支持资源和可复用指导。https://developers.openai.com/api/docs/guides/tools-skills
- OpenAI Build skills：技能目录、发现与加载机制。https://learn.chatgpt.com/docs/build-skills
- OpenAI MCP server：工具、资源、提示及结构化接口。https://developers.openai.com/plugins/concepts/mcp-server
- OpenAI Retrieval：搜索结果与原问题一起交给模型生成回答。https://developers.openai.com/api/docs/guides/retrieval
- OpenAI Agents API Architecture：托管 harness、模型与工具循环。这是特定产品架构；本方案使用自建 harness，不要求切换到该产品。https://developers.openai.com/api/docs/guides/agents-api/architecture

基准 notebook SHA-256：`326e23a73da70d090e0bd0865d5e55d40dc6df73916008133283862870f9448a`
