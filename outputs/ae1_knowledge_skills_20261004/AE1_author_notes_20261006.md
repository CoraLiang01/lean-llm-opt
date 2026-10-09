# AE1 回复核对说明（供作者使用，不属于正式回复）

本次在当前已打开的 AE1_response_document.tex 中原位更新完整正式回复，所有答复用蓝色标注，保留 AE 原评论与完整参考文献。另同步 AE1_response_blue.tex 与 AE1_references.bib。

## 核心表述

“According to the definitions in OpenAI's official documentation, RAG and Agent Skills are distinct concepts.” 是依据两组官方定义作出的归纳，而非官网逐字说“RAG is not a skill”的引文。

- RAG：检索内容扩充生成上下文。
- Agent Skill：可复用任务指令与支持资源的组织。
- 二者可组合：官方 skills 页面有 search/fetch 研究工作流例子。
- skill-like：描述 LEAN 固定程序指导的功能联系，不声称当前代码已实现 SKILL.md 包。
- 包装与学习是不同维度：技能不必通过学习获得，也不必评估中持续变化。

官方依据：
- https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy
- https://developers.openai.com/api/docs/guides/tools-skills
- https://developers.openai.com/plugins/concepts/skills

## 与代码对应

参照 LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb：参考案例库与路由指导固定；build_formulation_examples 用检索结果组装演示；NRM 的 CSVQA 内部计划由模型产生、由 Python 执行；标准其他路线以全部源行提供上下文进行提取和源行校验；Others 用 schema 和参考案例生成计划与代码。

正式回复没有把所有 CSVQA 都称为向量检索，没有声称所有路线只用一个案例或调用工具恰好一次，也没有声明自动学习或更新技能库。

## 稿件一致性

回复仅列作者已确认新增的两段：Section 1.2 的 Knowledge and Skills in LLM Agents，以及 Section 3.3 的 Connecting Reference Knowledge and Modeling Skills。

读取到的旧稿 /Users/cora/Downloads/main (3).tex 第421行仍描述 CSVQA 逐行向量检索；与0927代码的现行实现不同。该文件第366行的“three LLM agents”也不宜解读为标准路线的演示组装必然新增一次独立模型调用，当前组装由代码完成。请将这些实现说明与最终提交稿对齐。本次未修改 main (3).tex，也未声称这些修改已经完成。

## 验证

- 9个引用键均有独立文档内的 bibliography 条目与 BibTeX 条目。
- 当前 LaTeX 源文件已通过桌面编辑器编译。
- 没有运行付费模型或实验；本文不新增实验结论。
- OpenAI 文档日期为2026年10月6日的查阅日期，文档引用的2026不代表经核实的首次发布日期。
