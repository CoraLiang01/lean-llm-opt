# GPT-4.1 Model Interface V2 修改与验证报告

日期：2026-10-04。三份原 notebook 和历史结果未修改。新副本禁用默认整批运行，并使用独立输出目录。未修改当前审稿回复 LaTeX。

## 三份副本

- [LEAN_LLM_OPT_4.1_Large-scale_Model_Interface_V2.ipynb](../../LEAN_LLM_OPT_4.1_Large-scale_Model_Interface_V2.ipynb)；输出：`outputs/gpt41_model_interface_v2/full`。
- [LOTO_Examples_Only_GPT4.1_Large-scale_Model_Interface_V2.ipynb](../../LOTO_Examples_Only_GPT4.1_Large-scale_Model_Interface_V2.ipynb)；输出：`outputs/gpt41_model_interface_v2/examples_only`。
- [LOTO_Examples_And_Route_GPT4.1_Large-scale_Model_Interface_V2.ipynb](../../LOTO_Examples_And_Route_GPT4.1_Large-scale_Model_Interface_V2.ipynb)；输出：`outputs/gpt41_model_interface_v2/examples_and_route`。

## 修改范围

1. 各代码生成入口统一加入 MODEL_RETURN_CONTRACT，明确创建一个模型、调用优化、返回并暴露模型对象；Others 自有代码生成入口也同步。原提示已经要求返回，本次加强明确约定并用执行适配器兜底，而非只重复要求。
2. 执行器通过 AST 包装静态识别的 gurobipy.Model 构造调用，保留已创建的对象。优先识别全局 m/model，否则使用唯一全局模型或唯一捕获对象。仅有一个模型且其最优状态可读时才提取结果。
3. 函数未返回模型时可恢复其对象；不添加约束、不改目标、不注入 return、不自动调用未执行的函数，也不额外调用 optimize。多个模型、未优化、已释放或不存在模型均明确失败。动态工厂/动态导入等无法静态识别且未暴露全局模型的形式不保证可捕获。此机制不是安全沙箱。
4. 记录 selected_route、executed_route、route_allowed、模型对象选择方式、是否恢复、求解状态、执行成功标记、失败类别和源代码哈希。executed_route 表示通过守卫后进入的 workflow，不表示整条 workflow 成功。
5. 每题新增 generated_original.py、executed.py、execution.json，保留原 solve.py/model.md/solution.json。缓存验证覆盖新增产物，防止代码产物不一致被复用。
6. 所有修改同步到三个副本；变更缓存版本，LOTO 指向修改后的全量副本及其 SHA-256；原版文件哈希均保持不变。

分类器、允许标签规则、示例检索、各 route CSV 模式、建模提示、数据传递、变量域、求解器参数和 1e-4 评分容差未修改。没有增加 LLM 重试、类别纠正或代码重生成；model_contract_recovered 是执行接口恢复，不是额外推理。

## 已完成：冻结历史代码的真实求解复测

预先选定 12 份历史程序（跨三种配置，共 9 个不同题号），每份使用旧/新执行器各运行两次，共 48 次独立进程执行。生成程序及其内嵌数据不变，真实调用 Gurobi。没有重新调用 LLM，也没有以旧分数代替执行。参考目标仅用于执行后的评分。

样本选择包含四个缺少模型返回的失败（TP/NRM/RA/FLP 各一），两个正确对照、一个目标错误对照、一个未调用建模函数的负对照，以及全量/实验2的四个正常对照。它是诊断性抽样，不是随机代表性样本。

| 指标（每份程序计一次） | 原执行器 | 新执行器 |
|---|---:|---:|
| 成功提取最优解 | 7/12 | 11/12 |
| 目标值匹配 | 6/12 | 10/12 |

| 配置 | 题号 | 原执行器 | 新执行器 | 新目标 | 模型对象来源 |
|---|---|---|---|---:|---|
| examples_only | OR-004 | 失败 | 匹配 | 913520.1122932937 | captured_constructor |
| examples_only | OR-018 | 失败 | 匹配 | 782819438.64 | captured_constructor |
| examples_only | OR-036 | 失败 | 匹配 | 509925.0 | captured_constructor |
| examples_only | OR-063 | 失败 | 匹配 | 778095.4400000001 | captured_constructor |
| examples_only | OR-002 | 匹配 | 匹配 | 6464420.6613837 | global_m |
| examples_only | OR-069 | 匹配 | 匹配 | 4903.0 | global_m |
| examples_only | OR-001 | 求解但不匹配 | 求解但不匹配 | 125274.89905440263 | global_m |
| examples_only | OR-043 | 失败 | 失败 | — | 无模型 |
| examples_and_route | OR-011 | 匹配 | 匹配 | 3552260.54 | global_m |
| examples_and_route | OR-018 | 匹配 | 匹配 | 782819438.64 | global_m |
| full | OR-004 | 匹配 | 匹配 | 913520.1122996622 | global_m |
| full | OR-036 | 匹配 | 匹配 | 509925.0 | global_m |

四个接口失败样本 OR-004、OR-018、OR-036、OR-063 均恢复并匹配参考目标；原来成功的七份程序目标未变。OR-001 原本目标错误，仍然错误；OR-043 原本没有调用建模函数，仍明确失败。没有把原本错误的程序强行追认为正确。每个旧/新配对的两次重复在成功、目标匹配和错误状态上一致。

## 其他验证

- 12 项接口回归测试通过，使用真实小型 Gurobi 模型，覆盖无 return、别名导入、多个模型、未调用函数、未优化、不可行、已释放、运行异常及记录序列化。
- 三个副本在项目 lean_llm_opt_4_1 环境中加载定义成功；使用固定模拟分类/建模输出和真实求解器的 pipeline 集成检查通过。
- 三份副本的实际临时 CSV/产物保存、布尔值恢复、缓存复用、指纹变化和产物篡改拒绝检查通过。
- 实验2非法 route 在进入建模前被拦截，记录 selected_route 而 executed_route 为空；新执行器不绕过 LOTO 守卫。

## 未完成：从分类开始的真实 LLM 抽样

已尝试在项目环境启动六个预选题 OR-002、OR-004、OR-018、OR-036、OR-063、OR-069 的三条件抽样。读取 Test_Dataset/Large-scale-or/Large-scale-or-101.csv 时 PermissionError，预检查停止，未发生模型调用。参考库 CSV 在先前读取中也受保护。没有绕过文件保护，也没有用重建/替代数据冒充原输入。

因此现在只能确认执行接口在上述冻结程序样本上改善，不能宣称修改后的完整模型流程已达到 11/12，也不能将它外推到 101 题或改写 V1 的历史成绩。新增提示词对新生成代码的实际效果仍需可读取数据后的完整抽样。

## 复现入口

```bash
python scripts/test_gpt41_model_interface_v2.py
/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python scripts/validate_gpt41_interface_records.py
python scripts/sample_gpt41_model_interface_v2.py
```

数据恢复为可读取状态后，从仓库根目录运行完整抽样（会产生实际 API 调用）：

```bash
/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python scripts/run_gpt41_v2_end_to_end_sample.py --run
```

完整抽样每次在三个新配置目录下创建同一批次的 `sample_YYYYMMDD_HHMMSS` 子目录，不覆盖 V1 或上一批抽样，也不混入 101 题总成绩。脚本在完成抽样后自动调用 `scripts/compare_gpt41_v2_end_to_end_sample.py`，按题号与历史结果配对，生成 `END_TO_END_COMPARISON.md` 和 `end_to_end_comparison.json`。

目前历史六题基线为：全量 6 题求解、6 题匹配；Examples Only 2 题求解、2 题匹配；Examples And Route 6 题求解、6 题匹配。V2 三组均未产生真实推理记录，因此对照表标记为“待运行”，不会把 0 条记录误计为 0/6 成绩。当前详细状态见 `END_TO_END_COMPARISON.md`。

## 证据文件

- copy_manifest.json：原版/副本哈希及改动单元。
- sample_manifest.json：执行前固定的样本、输入代码及 notebook 哈希。
- replay_results.json、replay/：48 次实际执行的结果、日志与执行代码。
- interface_tests.log、pipeline_record_tests.json：回归和集成检查。
- end_to_end_sample_status.json：首次完整抽样预检查的阻塞记录（在随后诊断字段完善前产生，保留当时副本哈希）。
- END_TO_END_COMPARISON.md、end_to_end_comparison.json：六题×三条件的历史基线、V2 待运行状态及运行后配对成绩。
