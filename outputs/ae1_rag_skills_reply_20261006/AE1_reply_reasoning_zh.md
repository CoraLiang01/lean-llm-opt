# AE 回复建议：Knowledge、Skills 与 RAG

## 结论

OpenAI 的文档没有把 RAG 定义为 Skill。Skills 描述可复用程序指导及其资源如何组织，RAG 描述外部内容如何通过检索进入生成过程。二者可以组合，并非互斥的系统类型。

“RAG 可以组织进 skill”是根据官方技能结构和搜索工具工作流示例作出的设计推论，不是官方声明所有 RAG 都是 skill，更不是一个自动 merge API。技能可以包含说明、参考文件、脚本或指导工具使用；运行时检索内容可以成为其执行上下文。

## 对当前回复的修改理由

1. 增加明确的 RAG/Skills 关系，这是当前稿缺少的环节。
2. Ref-Data 不硬归为纯 declarative：案例既提供模型内容，也能通过演示传递程序经验。
3. 不把技能包装、技能学习称为同一种“狭义定义”。手工编写且评估时固定的技能仍然可以是技能。
4. 不说 LEAN 从案例中学习并持久化了 reusable skills：当前是固定指导与选定案例共同构成演示。
5. 不把参考案例检索等同于所有 CSV 数据操作。当前数据读取是另一种功能角色。
6. 使用 skill-like 描述方法上的联系；明确没有独立 SKILL.md 表示以及轨迹驱动技能库更新。
7. 回答 AE 对 agent/fine-tuning 的定位：指导被 agent 应用，也可与微调模型共用；不作普遍优劣判断。
8. 只列作者已说明的两处正文修改，不声称增加实验、实现 MCP 或完成技能重构。

## 官方文档证据

| 来源 | 文档内容 | 对本回复的支持 |
|---|---|---|
| OpenAI Skills | SKILL.md 指令及支持文件；references、scripts、assets | 技能是可复用指导与资源的模块化表示，不等同于案例搜索 |
| OpenAI Plugins Skills | 技能指导何时调用工具、顺序、处理不完整结果和最终输出；含 search/fetch 研究工作流示例 | 技能可以组织检索及证据使用流程 |
| OpenAI Optimizing LLM Accuracy | RAG 通过检索补充模型上下文；讨论 RAG 与 fine-tuning 的配合 | RAG 是内容访问与生成机制；不构成 fine-tuning 的普遍替代 |

链接：
- https://developers.openai.com/api/docs/guides/tools-skills
- https://developers.openai.com/plugins/concepts/skills
- https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy

## 论文与使用方式

| 文献 | 主要内容 | 在回复中的作用及边界 |
|---|---|---|
| Li et al., LREC-COLING 2024 | 分解并评估 declarative/procedural knowledge；前者是解题事实，后者是策略 | 支撑概念区分，不能由其定义证明每个 Ref-Data 案例都是纯事实 |
| Lewis et al., NeurIPS 2020 | 将参数模型与可检索的非参数记忆结合生成，并包含训练 | 支撑 RAG 机制；不把 LEAN 说成复现其完整训练架构 |
| Zhou et al., 2026 survey v3 | 技能为可复用程序制品，形式化为指导、辅助资源与适用条件；讨论表示/获取/检索/演化 | 支撑功能性 skill-like 联系，四阶段不是每个系统都必须具备 |
| Wang et al., ICML 2025 AWM | 从已有经历或在线交互中归纳工作流，供后续任务使用 | 对比技能/工作流从轨迹中获得的机制 |
| Yang et al., 2026 OptSkills v2 | 按问题结构原型聚类，蒸馏成功建模求解轨迹，细化或扩展技能库 | 同领域最直接的联系；LEAN 的固定整理方式与其技能学习不同 |
| Rubin et al., NAACL 2022 EPR（可选） | 利用下游生成信号训练示例检索器，选择上下文参考例题 | 支撑相关例题应用，精简回复可不额外引用 |

论文来源：
- https://aclanthology.org/2024.lrec-main.980/
- https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
- https://arxiv.org/html/2605.07358v3
- https://proceedings.mlr.press/v267/wang25bx.html
- https://arxiv.org/abs/2605.29829v2
- https://aclanthology.org/2022.naacl-main.191/

## 0927 代码核对

- 使用统一参考 CSV；类别内 FAISS 查询返回案例，而不是技能包。
- build_formulation_examples 将选定案例、CSVQA 动作、数据 observation 与专家模型组装成演示。
- get_NRM_response 保留固定类别指令，要求符号模型与精确 Data Mapping。
- NRM CSVQA 生成提取计划再执行；其余标准 route 用全部源行进入上下文进行提取和校验。
- Others 使用当前 schema 与参考案例生成抽象计划，再生成读取完整 CSV 的程序。
- 有固定路由和指导，未见独立技能包注册/读取或从已完成轨迹更新技能库的流程。
- 参考案例中的数据用于演示；当前实例数据为当前目标模型提供参数。

代码 SHA-256（本次核对）：326e23a73da70d090e0bd0865d5e55d40dc6df73916008133283862870f9448a。

## 建议英文回复

以下使用已有 citation keys，并新增 OpenAI accuracy 文档条目。最后的正文修改说明依据作者此前报告的两处增补；本次没有修改 main.tex 或当前打开的回复文档。

We thank the Associate Editor for suggesting this perspective. We agree that the distinction between reference knowledge and category-specific procedural expertise clarifies the positioning of LEAN-LLM-OPT.

Prior work distinguishes declarative knowledge, such as facts relevant to a problem, from procedural knowledge, such as strategies for performing a task \citep{ae1_li2024metacognitive}. In LEAN-LLM-OPT, Ref-Data provides external, case-based modeling knowledge: reference problem descriptions, associated data, expert formulations, and annotations identifying relevant data. These cases illustrate model structure and the correspondence between data and model components. They can also convey procedural information when incorporated into a worked demonstration; we therefore do not regard reference knowledge as exclusively declarative.

Retrieval-augmented generation (RAG) describes how external content is retrieved and used to condition generation \citep{ae1_lewis2020rag}. OpenAI's documentation likewise distinguishes this mechanism from Agent Skills: RAG supplies relevant context, whereas a skill packages reusable instructions and supporting resources for carrying out a task \citep{ae1_openai2026accuracy,ae1_openai2026skills}. These concepts concern different aspects of a system and can be combined. A skill can prescribe how to search for, interpret, and apply retrieved material; OpenAI explicitly illustrates skills that combine search and fetch tools into a research workflow \citep{ae1_openai2026skilltools}. Thus, a skill may organize a RAG workflow, but retrieving reference content does not by itself constitute a modeling skill.

In LEAN-LLM-OPT, reference retrieval selects relevant cases, while category-conditioned instructions specify how the agent should use them together with the current problem and its data. For the predefined categories, selected cases are assembled into demonstrations of data access and formulation. The model-generation agent then follows this guidance to obtain instance data and construct the target model. For example, the NRM route combines a reference demonstration with instructions for obtaining data, defining a symbolic formulation, and specifying exact data-to-model mappings. The general route instead constructs an instance-specific abstract modeling plan from the current query and data schema. Reference cases provide modeling examples; target datasets provide the parameters of the current instance.

This reusable procedural component is \emph{skill-like}, consistent with the characterization of skills as procedural artifacts with instructions, supporting resources, and applicability conditions \citep{ae1_zhou2026skillsurvey}. Reuse concerns the modeling method, not an identical action sequence or output across instances. Our implementation embeds this method in prompts, workflow-construction code, and data tools rather than standalone \texttt{SKILL.md} packages. Separately, it does not acquire or revise a persistent skill library from completed trajectories. This distinguishes its design-time curation from the workflow induction studied in Agent Workflow Memory and the archetype-based skill distillation and refinement studied in OptSkills \citep{ae1_wang2025awm,ae1_yang2026optskills}.

Accordingly, we position LEAN-LLM-OPT as a \emph{reference-guided agentic workflow construction framework} that combines case-based modeling knowledge with reusable, category-conditioned procedural guidance for optimization modeling with external data. The agents apply these resources through reasoning and tool use; the resources could also be used with a fine-tuned model. Our contribution concerns their organization and application at inference time, rather than establishing that references replace agents or are universally preferable to fine-tuning.

We have added \emph{Knowledge and Skills in LLM Agents} to Section~1.2 (\emph{Related Works}) to explain knowledge, RAG, and skills and their connections to recent literature. We have also added \emph{Connecting Reference Knowledge and Modeling Skills} to Section~3.3 (\emph{Novelties in Our Design and Key Messages}) to explain their roles in LEAN-LLM-OPT and distinguish procedural reuse from standalone skill packaging and trajectory-based skill learning.

## 正文两处应同步强调

Section 1.2：定义 reference knowledge、RAG、skills；说明技能可以使用检索，而检索机制不限于事实类内容。

Section 3.3：对应实际框架，说明固定类别指导与动态案例演示如何结合；区分参考与目标数据；说明现有方法的 skill-like 性质及包装/学习范围。

可增补一句：

> RAG and skills concern different aspects of the framework: retrieval supplies relevant reference content, whereas reusable procedural guidance specifies how the agents use that content and the target data to construct a formulation.

这是一项概念与方法定位澄清，并不证明各组件的因果贡献；若另有机制贡献断言，需要相应实验支持。
