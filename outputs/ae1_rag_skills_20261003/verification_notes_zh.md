# AE1：RAG、knowledge 与 skills 的定位核查

核查日期：2026-10-03。本文为作者核查说明；AE1_response_blue.tex 是回复片段，AE1_references.bib 是配套引用。

## 结论

RAG 是检索与生成相结合的机制；知识是其所提供的内容；skill 是组织可复用做法的程序性对象。三者不能作为互斥标签处理。事实、案例、流程、模板都可能成为检索对象；一个 skill 可以包含知识资料，也可以指导执行时的检索。

“merge”应区分以下含义：

| 含义 | 核查结论 |
| --- | --- |
| 将整理过的参考案例和程序说明放在同一 skill 中 | 官方支持 instructions + references 等辅助文件。 |
| skill 指导检索，并利用返回资料完成任务 | 官方明确举例搜索/获取工具与程序性工作流结合；将此用于 RAG 是有依据的设计推论。 |
| 将选定参考内容与建模指令放入同一 prompt/示范 | 与 LEAN 正文第 3.2.1 节及当前 get_TP_response 的拼装过程一致；不等于已经采用 SKILL.md 封装。 |
| 单纯拼接文本就自动获得一个有效 skill | 没有这种定义依据；需要跨实例适用的程序、适用范围及资料使用方式。 |
| OpenAI 存在通用 RAG-to-skill 自动 merge 功能 | 所查页面未给出此功能。不能将“允许组合”表述为“提供自动转换保证”。 |

不应再用“不能自学习”作为“不是 skill”的理由；应分别说明表示方式、执行时内容的变化，以及是否有跨任务学习。

## 文献与官方证据

1. **Lewis et al., NeurIPS 2020, RAG**：将参数化模型与检索得到的非参数资料结合，支持把 RAG 定位为访问和使用外部内容的机制。原论文不提供现代 agent skill 的定义。[论文](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
2. **Li et al., LREC-COLING 2024**：第 3.1 节区分事实性内容与一般解题策略。支持角色区分，不能单独证明 Ref-Data 是纯陈述性知识。[论文](https://aclanthology.org/2024.lrec-main.980.pdf)
3. **Buffer of Thoughts, NeurIPS 2024**：存储、检索并实例化解题模板。说明检索对象可以具有程序性内容；不应将其说成复现原始 RAG 架构。[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cde328b7bf6358f5ebb91fe9c539745e-Abstract-Conference.html)
4. **Zhou et al., 2026, v3 综述**：第 II-D 节的 skill 对象包含指令、辅助资源和适用条件；第 IV-A 节包含人工来源。可用于功能映射，不宜把它与所有技能包文献强行分成两套公认定义。[原文](https://arxiv.org/html/2605.07358v3)
5. **OpenAI API Skills 文档**：What’s a skill 部分说明目录、SKILL.md 和辅助资料；references 用于背景资料。它是官方产品定义而非学术有效性证明。[官方文档](https://developers.openai.com/api/docs/guides/tools-skills)
6. **OpenAI Plugins Skills 文档**：How skills complement an MCP server 部分明确给出搜索/获取工具及返回信息与工作流结合的例子。[官方文档](https://developers.openai.com/plugins/concepts/skills)
7. **OpenAI Build skills 文档**：Add supporting resources 部分建议把详细案例、schema、背景资料放在 references 并说明何时读取；组合不要求把整个知识库塞入 SKILL.md。[官方文档](https://developers.openai.com/plugins/build/skills)
8. **SkillsBench v4**：附录 M.1 是基准构建的操作性定义，要求程序性内容、类别适用性、文件结构和可移植性。它对普通 RAG retrievals 的排除，不应扩展成“skill 不能使用检索”，其辅助资源本身允许 references 和 worked examples。[原文](https://arxiv.org/html/2602.12670v4)
9. **AWM / OptSkills**：分别用于对照从经历归纳工作流、从优化轨迹提炼并维护类别技能的学习机制。人工编写且评估时不更新的程序仍可具有 skill 属性。[AWM](https://proceedings.mlr.press/v267/wang25bx.html)；[OptSkills](https://arxiv.org/html/2605.29829v2)

## LEAN 与正文的对应

已读取 `/Users/cora/Downloads/main (3).tex` 第 3.2 节及当前仓库 `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb` 的 get_TP_response。正文中 workflow generation agent 将 q、g、f、m 嵌入示范步骤，下游 model generation agent 再根据目标数据建模。对应代码将参考问题、数据、模型插入 few_shot_examples 的程序性模板。故“联合表示已经存在”有实现依据；“已实现 OpenAI skill 封装”没有。

此前引用的 V2 notebook 和 20260919 输出目录在当前工作区不存在，本次未把它们当作重新检查过的文件。当前回复依据用户粘贴文本、可读取的 main (3).tex 和上述现存 notebook；不涉及重做实验或判断尚未检查的最新运行版本。

用户确认已新增的正文部分只有第 1.2 节 Knowledge and Skills in LLM Agents，以及第 3.3 节 Connecting Reference Knowledge and Modeling Skills。回复末段仅保留这两处，不声称修改了摘要、贡献或实验。

为使正文与新增 RAG 论证一致，建议在这两处现有新增段落分别补充一句：

- 第 1.2 节：`Retrieval determines how external material is accessed, whereas skills organize how it is used; reference knowledge can therefore support a reusable procedure, either as packaged material or as information retrieved during execution.` 可引 Lewis、Zhou 和 OpenAI 文档。
- 第 3.3 节：`In the type-tailored workflow, reference content and procedural instructions are jointly represented in the constructed demonstration; their distinction concerns their roles rather than separate storage or presentation.`

以上两句是建议补充，未写入原稿。所有提供的 TeX 为回复片段，未声称整封 reply letter 或论文已编译。
