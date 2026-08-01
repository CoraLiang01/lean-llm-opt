# 实验配置统一 + 运行日志规范 —— 改造指引

对应审稿意见：**Comment 3.2**（baseline 公平性、组件对照表）、**Comment 4.1**（token / LLM 调用次数 / tool 调用次数 / latency / 美元成本，形成 cost–performance 对照）、以及 Comment 4.2 蒸馏讨论所需的 workflow traces。

改造后所有 notebook 共用一份 `exp_config.yaml`，所有 LLM 从同一个工厂函数产生，所有调用自动被计量，每题一行 JSONL 日志。**建模逻辑一行不用动**，只动"造模型"和"跑循环"两处。

---

## 0. 当前状态：改造已执行完毕

7 个 notebook 已经全部改好。改造后的版本**沿用原文件名放在仓库根目录**，改造前的原件
原封不动地移到了 `original_notebooks/`。

```
lean-llm-opt-main/
├── leanopt_exp.py                    ← 新增：配置 + 计量回调 + 日志
├── exp_config.yaml                   ← 新增：唯一配置源
├── aggregate_runs.py                 ← 新增：汇总出表
├── test_smoke.py                     ← 新增：离线自检
├── apply_instrumentation_patch.py    ← 新增：执行改造的脚本（可重跑）
├── runs/                             ← 跑起来后自动生成
│
├── LEAN_LLM_OPT_4.1_Air_NRM.ipynb                 ← 改造后，跑这些
├── LEAN_LLM_OPT_4.1_Large-scale-or.ipynb          ← 改造后
├── LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb         ← 改造后
├── LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb  ← 改造后
├── Ablation_Study_Air_NRM_RAG_Only.ipynb          ← 改造后
├── Ablation_Study_Air_NRM_Few-shot_Only.ipynb     ← 改造后
├── Benchmark_Base_Model_Small_Scale.ipynb         ← 改造后
│
├── original_notebooks/               ← 改造前的原件，7 个，一字未动，不要跑
│
├── Test_Dataset/  Large_Scale_Or_Files/  output/  run_all_*.py   ← 未改动
```

> 根目录那 7 个是**生成物**：`original_notebooks/` + `apply_instrumentation_patch.py`
> 才是源头。手工改了根目录的 notebook 之后再跑改造脚本，改动会被覆盖 ——
> 要么把改动写进脚本的规则里，要么之后别再跑它。

`apply_instrumentation_patch.py` 是可重跑的：每条替换规则都声明了预期匹配次数，对不上就中止而不是产出半成品。想调整规则后重新生成，直接 `python apply_instrumentation_patch.py`（`--check` 是空跑）。

### 改造后每个 notebook 的实际改动量

| notebook | build_llm | embeddings | retriever | agent 参数 | 主循环 | 输出路径 |
|---|---|---|---|---|---|---|
| 4.1_Air_NRM | 4 | 5 | 9 | 3 | Process_Input + Batch | 8 |
| oss_Air_NRM | 1 (shim) | 1 | 9 | 3 | Process_Input + Batch | 12 |
| 4.1_Large-scale-or | 9 | 13 | 12 | 7 | run_test | 32 |
| oss_Large-scale-or | 1 (shim) | 2 | 5 | 6 | run_test | 25 |
| Ablation_RAG_Only | 3 | 5 | 9 | 3 | Process_Input + Batch | 8 |
| Ablation_Few-shot_Only | 3 | 1 | 1 | 3 | Process_Input + Batch | 8 |
| Benchmark_Base_Model | shim | 0 | 0 | 0 | run_test | 8 |

### 已通过的验证

- 每个改造后 notebook 的所有 code cell 通过 `ast.parse`
- 改造后不再残留 `ChatOpenAI(` / `ChatOllama(` / `OpenAIEmbeddings(` / `openai_api_key=user_api_key` / `time.sleep(15)`
- 7 套 `model_profile` / `method` / `dataset` 组合都能成功 `load_config`
- 用桩函数实跑了改写后的 `run_test` 和 `Batch_Process_Queries`：正常路径长度对齐、异常路径长度也对齐且状态记为 `failed`
- 用真实 `langchain-core 1.2.7` 跑通回调链路（LLM 调用 / tool 调用 / 阶段归属 / token 兜底估算）
- `python test_smoke.py` 通过

### 改造过程中发现并修掉的三个问题

1. **`except: continue` 导致结果错位**——失败样本不 append，后续 `pd.DataFrame({'Query': test['Query'], 'model_output': output_model})` 按位置拼接，query 与 output 对不上号。现在失败也 append `None`。
2. **`Process_Input` 的 `UnboundLocalError`**——分类结果不含 "Sales-Based Linear Programming" 时 `Type` / `output_model` 未绑定，抛的是 UnboundLocalError 而不是可读的错误。现在显式 `raise ValueError("unroutable problem type: ...")`，并记进日志。
3. **`agent_max_iterations` 口径不一**——4.1 系列无上限，oss 系列是 5。统一到 5（写在 `exp_config.yaml`）。这会让 4.1 的行为发生变化，属于必要的统一。

### 还需要你手动补的

- **`exp_config.yaml` 里 `common.hardware_note` 是 TODO**，填上跑 gpt-oss-20b 的显卡型号和 Ollama 版本，本地模型的成本口径要靠它。
- **`gold_type` 列**：`run_test` 会读 `row.get('Type')` 当真值标签；如果测试集里这列叫别的名字，改一下 §4 里那一行。
- **删掉自检产物** `_smoke_runs/`、`_smoke_tables/`、`_smoke_gold.csv`（我这边没有删除权限）。
- **注释掉的旧版本 cell 没删**（你没勾选这项）：`LEAN_LLM_OPT_4.1_Air_NRM.ipynb` 末尾两个 `_V1`/`_V2` 死代码，以及 benchmark notebook 里三份注释切换的 `build_llm` 已经被新的 shim 替换掉了。
- **改造后 notebook 的 cell 输出已清空**（旧输出来自改造前的代码，留着会误导）。

---

## 1. 当前各 notebook 的配置不一致清单

改之前先看这张表，这几条正是审稿人追问 baseline 公平性时会被抓住的：

| 项目 | LEAN 主实验 (4.1) | Ablation RAG-Only | Ablation Few-shot-Only | Benchmark Base Model | LEAN 主实验 (oss) |
|---|---|---|---|---|---|
| 分类 agent 模型 | **`gpt-4`** | **`gpt-4.1`** | **`gpt-4.1`** | 无分类环节 | gpt-oss:20b |
| 建模 / 代码模型 | `gpt-4.1` | `gpt-4.1` | `gpt-4.1` | 注释切换 | gpt-oss:20b |
| 解码参数 | 部分 `top_p=1,n=1`，部分只有 `temperature` | 只有 `temperature` | 只有 `temperature` | GPT-5 用 `temperature=1.0`，其余 `0.0` | 只有 `temperature` |
| 模型快照 | 别名，未锁日期 | 别名 | 别名 | 别名 | — |
| embedding | OpenAI 默认（未显式指定型号） | 同左 | 同左 | `langchain.embeddings.openai`（已废弃路径） | `nomic-embed-text` |
| 检索 k | 混用 5 / 1 / `max_tokens_limit=400` | 5 / 1 | 5 / 1 | — | 混用 |
| agent 迭代上限 | 无 | 无 | 无 | — | 无 |
| 随机种子 | 无 | 无 | 无 | 无 | 无 |

三个问题按严重程度排：

**(a) 主实验分类用 `gpt-4`，消融实验分类用 `gpt-4.1`。** 这直接影响 Comment 3.2 —— 消融对比本应只差"有没有 workflow"，现在还差了一个分类模型，差异归因不干净。而且论文正文写的是 GPT-4.1，审稿人查代码会发现分类环节是另一个模型。

**(b) `Benchmark_Base_Model_Small_Scale.ipynb` 用注释掉/取消注释来切换模型**（cell 2 里 `build_llm` 有三个版本，gpt-oss:20b 生效，gpt-4.1 和 gpt-5 被注释）。这意味着从代码无法判断某个结果 CSV 是哪个模型跑的——`MAMOc_1-25_5.2.csv` 这样的文件名是唯一线索。审稿人要求 report exact model versions 时这是答不上来的。

**(c) GPT-5 系列用 `temperature=1.0`，其他模型用 `0.0`。** 如果是因为端点不接受 0.0，需要在论文里明说；如果只是历史遗留，应统一。

改成统一 profile 之后，日志里记的是实际返回的模型名和 `system_fingerprint`，不会再出现代码与论文对不上的情况。

---

## 2. 通用改造：所有 notebook 都一样的三处

### 2.1 首个 import cell —— 追加

**删掉**这一行（明文 key 不该留在 notebook 里）：

```python
user_api_key = "Your OpenAI API Key"
```

**换成**：

```python
import os, leanopt_exp as lx

os.environ.setdefault("OPENAI_API_KEY", "")   # 从环境变量读
user_api_key = os.environ["OPENAI_API_KEY"]   # 兼容仍在引用它的旧代码

CFG = lx.load_config(
    "exp_config.yaml",
    model_profile="gpt-4.1",          # oss 版本改 "gpt-oss-20b"
    method="LEAN-LLM-OPT",            # 消融改 "Abl-RAGOnly" 等
    dataset="Large-Scale-OR",         # 见 exp_config.yaml 的 datasets 段
    repeat_index=0,                   # 重复实验时 0/1/2
)
LOG = lx.RunLogger(CFG)
EMBEDDINGS = lx.build_embeddings(CFG)
```

各 notebook 对应的 `model_profile` / `method` / `dataset` 取值见 §3 每小节开头。

### 2.2 所有 `ChatOpenAI(...)` / `build_llm(...)` —— 换成工厂

| 原写法 | 改成 |
|---|---|
| `llm1 = ChatOpenAI(temperature=0.0, model_name="gpt-4", openai_api_key=user_api_key)` | `llm1 = lx.build_llm(CFG, "classifier")` |
| `llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4.1', top_p=1, n=1, ...)` | `llm2 = lx.build_llm(CFG, "modeler")` |
| `llm = ChatOpenAI(model="gpt-4.1", temperature=0, ...)` | `llm = lx.build_llm(CFG, "data_agent")` |
| `llm_code = ChatOpenAI(temperature=0.0, model_name="gpt-4.1", ...)` | `llm_code = lx.build_llm(CFG, "coder")` |
| `build_llm(model="gpt-oss:20b", temperature=0.0)` | `lx.build_llm(CFG, "modeler")` |

`OpenAIEmbeddings(openai_api_key=user_api_key)` / `OllamaEmbeddings(model="nomic-embed-text")` 全换成 `EMBEDDINGS`。

> 计量回调是在 `build_llm` 里挂上去的，**换完这一步 token / 成本 / 延迟就自动在记了**，agent 内部套多少层都跑不掉。

### 2.3 检索器与 agent —— 统一参数

```python
# 原：vectors.as_retriever(search_kwargs={'k': 5})
retriever = lx.build_retriever(CFG, vectors, "refdata")

# 原：vectors.as_retriever(max_tokens_limit=400, search_kwargs={'k': 1})
retriever = lx.build_retriever(CFG, vectors, "data")

# 带 filter 的（Air_NRM 里按 OD/time 过滤）
retriever = lx.build_retriever(CFG, new_vectors, "data",
                               search_kwargs={"filter": {"OD": od, "time": time}})

# 原 initialize_agent(...)
agent = initialize_agent(
    tools=[qa_tool], llm=llm2,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    **lx.agent_kwargs(CFG, prefix, suffix),
)
```

`lx.agent_kwargs` 会补上原来缺失的 `max_iterations`、`max_execution_time`、`early_stopping_method`。加上限不只是为了统一——ReAct agent 卡住时会无限循环烧 token，成本表里会出现无法解释的长尾，审稿人问"为什么这题花了 50 次调用"时没法解释。

---

## 3. 逐 notebook 的具体位置

### 3.1 `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb`
`model_profile="gpt-4.1"`, `method="LEAN-LLM-OPT"`, `dataset="Large-Scale-OR"`

| cell | 原内容 | 操作 |
|---|---|---|
| 1 | import + `user_api_key` | §2.1 |
| 4 | `llm1`(**gpt-4**) + `classification_agent` | `build_llm(CFG,"classifier")`；retriever 用 `"refdata"`；agent 用 `lx.agent_kwargs` |
| 7, 9, 11, 13, 15 | `get_NRM/RA/TP/AP/FLP_response` | 每个 cell 改 4 处：`llm2`→`modeler`、embeddings→`EMBEDDINGS`、retriever→`build_retriever(CFG,v,"data")`、agent→`**lx.agent_kwargs(...)` |
| 17 | `get_Others_response`（含两条 chain） | 同上，两条 chain 的 llm 也走 `modeler` |
| 19 | `get_others_without_CSV_response` | 同上 |
| 21 | `llm_code` + `get_code` | `build_llm(CFG,"coder")` |
| 23 | `run_test` | 按 §4 重写循环 |
| 28–33 | 分片跑 + `read_and_combine_csvs` | 输出改到 `LOG.dir`，见 §5 |
| 34–72 | NL4OPT / IndustryOR / MAMO 各段 | 同 28–33，只改 `dataset=` 和输出路径 |

### 3.2 `LEAN_LLM_OPT_4.1_Air_NRM.ipynb`
`model_profile="gpt-4.1"`, `method="LEAN-LLM-OPT"`, `dataset="Air-NRM-CA"` / `"Air-NRM-NP"`

| cell | 原内容 | 操作 |
|---|---|---|
| 0 | import + `user_api_key` | §2.1 |
| 2 | `Classification_Agent`，`llm1=gpt-4`，`k=5` | §2.2 + §2.3 |
| 6 | `New_Vectors_Flight` / `New_Vectors_Demand` 的 embeddings | 换 `EMBEDDINGS`。顺带清理：`retrieve_parameter` 里 `v1,v2,df,demand = LoadFiles()` 的解包顺序与 `LoadFiles` 返回的 `v1,v2,demand,flight` 对不上（`df` 实为 demand、`demand` 实为 flight）。当前不影响结果（这两个变量函数体里没用到），但每次调用都重读四个 CSV，建议只解包 `v1, v2` 并加缓存 |
| 8 | `csv_qa_tool_flow` + `llm` + `llm_code` + `agent2` | `llm`→`data_agent`，`llm_code`→`coder` |
| 10 | `csv_qa_tool_CA` + `llm` + `agent2` | 同上 |
| 12 | `Process_Input` | 加 stage 标注，见 §4 |
| 14 | `Batch_Process_Queries` | 按 §4 重写 |
| 17, 19 | 跑 CA / NP_Flow + 抽代码 + `gain_obj` | 输出改到 `LOG.dir`；`gain_obj` 加 TimeLimit，见 §6 |
| 20, 21 | 整段注释掉的 `_V1` / `_V2` 旧版本 | **删掉**。留在提交材料里容易被读成结果是挑出来的 |

### 3.3 `LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb`
`model_profile="gpt-oss-20b"`, `method="LEAN-LLM-OPT"`, `dataset="Large-Scale-OR"`

| cell | 原内容 | 操作 |
|---|---|---|
| 0 | import | §2.1 |
| 3 | `build_llm(model="gpt-oss:20b")` + `OllamaEmbeddings` | **整个 cell 删掉**，改用 `lx.build_llm` / `lx.build_embeddings` |
| 4 | `get_code` | llm→`coder` |
| 6 | 分类 + few-shot + 关键词抽取（`llm1`, `llm2`） | `llm1`→`classifier`，`llm2`→`modeler` |
| 10,12,14,16,18 | 各类型 response 函数 | 同 §3.1 |
| 20 | `initialize_abstract_modeler_chain` / `initialize_code_gen_chain` | →`modeler` / `coder` |
| 22 | `get_others_without_CSV_response` | 同上 |
| 24 | `run_test` | §4 |

### 3.4 `LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb`
`model_profile="gpt-oss-20b"`, `dataset="Air-NRM-CA"` / `"Air-NRM-NP"`

结构与 3.2 对应：cell 0 配置、cell 2 分类 agent、cell 8/10 两条 flow、cell 12 `Process_Input`、cell 14 批处理。

### 3.5 `Ablation_Study_Air_NRM_RAG_Only.ipynb`
`model_profile="gpt-4.1"`, `method="Abl-RAGOnly"`, `dataset="Air-NRM-CA"` / `"Air-NRM-NP"`

| cell | 原内容 | 操作 |
|---|---|---|
| 2 | import + `user_api_key` | §2.1 |
| 6 | `llm1`(**gpt-4.1**) + embeddings + `agent_pc` | 换 `classifier` 角色——**这一步会把分类模型从 gpt-4.1 改成与主实验一致**，消融对比才干净 |
| 8 | 两处 embeddings | `EMBEDDINGS` |
| 9, 10 | `llm` + `agent2` | `data_agent` + `lx.agent_kwargs` |
| 11 | 主流程函数 | 加 stage 标注 |
| 15, 17 | 跑 CA / NP_Flow | §4 + §5 |

> 改完这个 notebook 要重跑消融结果。分类模型变了，旧数字不能直接用。

### 3.6 `Ablation_Study_Air_NRM_Few-shot_Only.ipynb`
`method="Abl-FewShotOnly"`，位置与 3.5 基本一致（cell 2 import、cell 5 `llm1`+embeddings）。

### 3.7 `Benchmark_Base_Model_Small_Scale.ipynb`
`method="Base-SingleCall"`，`model_profile` 按跑的模型切换

这个 notebook 改动最大，因为它现在靠注释切换模型：

| cell | 原内容 | 操作 |
|---|---|---|
| 0 | `from langchain.embeddings.openai import OpenAIEmbeddings` | 已废弃路径，换 `EMBEDDINGS` |
| 2 | `build_llm` 三个版本（gpt-oss:20b 生效，gpt-4.1 / gpt-5 被注释） | **三个全删**，改成 `llm = lx.build_llm(CFG, "modeler")`。跑哪个模型只改 cell 0 里的 `model_profile=` |
| 2 | `run_test(df, llm)` | §4 |
| 7–15 | 分片跑 MAMOc，输出 `MAMOc_*_5.2.csv` | 输出改到 `LOG.dir`；模型信息不再靠文件名区分，由 `run_id` 承载 |

要跑 Gemini 3 Pro 的话，`model_profile="gemini-3-pro"` 即可，`build_llm` 里已经接了 `ChatGoogleGenerativeAI`（`langchain-google-genai` 在 requirements 里）。

---

## 4. 主循环改造（关键一步）

所有 notebook 的主循环都改成这个模式。以 `run_test` 为例：

```python
def run_test(test, agent, log=LOG, cfg=CFG):
    output_model, output_code, classification = [], [], []

    for index, row in test.iterrows():
        query = row["Query"]
        with log.instance(instance_id=int(index),
                          query=query,
                          gold_type=row.get("Type"),          # 有真值标签就传
                          dataset_address=row.get("Dataset_address")) as rec:
            try:
                # ---- 阶段 1：问题分类 ----
                with lx.stage("classification"):
                    response = lx.call_with_retry(
                        agent.invoke, cfg,
                        f"What is the problem type of the text? text:{query}")
                selected_problem = extract_problem_type(response["output"])
                rec.set(pred_type=selected_problem)

                # ---- 阶段 2：类型专属工作流 + 数据检索 ----
                with lx.stage("modeling"):
                    if csv_detect(row):
                        dataset_address = row["Dataset_address"]
                        if selected_problem in ("Network Revenue Management", "NRM"):
                            output = get_NRM_response(query, dataset_address)
                        elif ...:
                            ...
                    else:
                        output = get_others_without_CSV_response(query)

                # ---- 阶段 3：代码生成 ----
                with lx.stage("codegen"):
                    code_response = get_code(output, selected_problem)

                rec.set(model_output=output, code_output=code_response)
                output_model.append(output)
                output_code.append(code_response)
                classification.append(selected_problem)

            except Exception as e:
                # 失败也要留一行，否则失败率无法统计
                rec.set(model_output=None, code_output=None)
                output_model.append(None)
                output_code.append(None)
                classification.append(None)
                print(f"[{index}] failed: {type(e).__name__}: {e}")

    return output_model, output_code, classification
```

要点：

1. **`with log.instance(...)` 是唯一必须加的包装**，它把当前题目设为"计量目标"，回调记录的一切都落到这一行。
2. **`lx.stage("...")` 决定成本能否拆到阶段**。不标也能跑，但汇总表里就只有总数、没有"分类占多少 / 检索占多少 / 建模占多少"。Comment 4.1 要的 cost–performance 对照、Comment 4.2 判断"哪一段值得蒸馏"，都靠这个拆分。四个阶段名统一用 `classification` / `data_retrieval` / `modeling` / `codegen`。Air_NRM 的 `csv_qa_tool_flow` 内部检索建议单独包 `data_retrieval`。
3. **异常不要 `continue` 掉**。原 `run_test` 里 `except requests.exceptions.RequestException: continue` 会让 `output_model` 和 `test` 行数对不上，后面 `pd.DataFrame({'Query': test_1['Query'], 'model_output': output_model})` 按位置拼接会直接错位；而且失败样本被静默丢弃等于准确率分母变小。这是当前数字可能不准的实际风险，优先改。
4. **删掉 `time.sleep(15)`**，改由 `cfg.sleep_between_instances_s` 控制（默认 0）。原来 101 题白等 25 分钟，且这段等待会污染 wall-clock 延迟统计。

跑完调用 `LOG.close()`，会打印并写入本次 run 的汇总。

---

## 5. 输出路径统一

原来的文件名（`Large-scale-or-Lean-4.1_1.csv`、`RAGOnly_CA_GPT4.1_bench_New.csv`、`MAMOc_1-25_5.2.csv`、`OBJ_NP_New_4.1_V2.csv`）无法反查是哪套配置跑的。改成：

```python
output_df.to_csv(LOG.dir / "predictions.csv", index=False)
```

`run_id` 形如 `LEAN-LLM-OPT__gpt-4.1__Large-Scale-OR__r0__20260724-165525__2fca70a7cf0e`，末段是配置哈希；同目录的 `config.json` 存了完整配置 + 环境清单（包版本、Gurobi 版本、主机名、时间戳）。分片跑没问题，同一个 `LOG` 对象往同一个 `instances.jsonl` 追加。

---

## 6. 求解环节也要计时和限时

`gain_obj` / `run_gurobi_code` 现在没有时间上限，且求解时间没进报告。建议：

```python
def run_gurobi_code(code_str, cfg=CFG):
    t0 = time.perf_counter()
    ...
    env["__time_limit__"] = cfg.solver_time_limit_s
    code_str += "\nif 'm' in dir():\n    m.Params.TimeLimit = __time_limit__\n"
    ...
    rec = lx._CURRENT_RECORD.get()
    if rec is not None:
        rec.set(solve_time_s=round(time.perf_counter() - t0, 3),
                n_vars=getattr(env.get("m", None), "NumVars", None),
                n_constrs=getattr(env.get("m", None), "NumConstrs", None),
                n_nonzeros=getattr(env.get("m", None), "NumNZs", None))
```

顺带解决 **Comment 4.1 的前半段**——"按问题类别统计变量数、约束数、非零元数量的 min/median/max"。这三个数记进日志后汇总脚本可以直接出表，不用另写脚本。

---

## 7. 本地模型的成本口径

gpt-oss-20b 的美元成本是 0，但不能说它"免费"。日志里已有 wall-clock，另外把 `exp_config.yaml` 里的 `common.hardware_note` 填上。论文对比时用两列口径：API 模型报 USD/instance，本地模型报 GPU-seconds/instance，不要混成一个数。

---

## 8. 跑完出表

```bash
python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv
```

`gold_labels.csv` 格式（人工判定放这里，与运行日志分离，重跑不会覆盖标注）：

```csv
dataset,instance_id,gold_type,correct_optimal,correct_formulation
Large-Scale-OR,0,NRM,1,1
Large-scale-OR,1,RA,1,0
```

产出：

| 文件 | 内容 | 对应 comment |
|---|---|---|
| `cost_table.csv` / `.tex` | 每个 (方法, 模型, 数据集) 的准确率、LLM 调用数、tool 调用数、prompt/completion token、USD、延迟 mean/p95、失败率 | 4.1 |
| `stage_breakdown.csv` | 上述指标按 classification / data_retrieval / modeling / codegen 拆分 | 4.1、4.2 |
| `classification.csv` + `confusion__*.csv` | 分类准确率、混淆矩阵、**分类正确 vs 错误条件下的建模准确率** | 3.1 |
| `per_instance.csv` | 每题一行，画成本–准确率散点图用 | — |

`cost_table.tex` 可直接 `\input` 进论文。

---

## 9. 建议的实验矩阵

配置好之后，同一套代码只改 `model_profile` 和 `method` 两个字段就能跑完下面这些格子。

| method | model_profile | 状态 | 对应 comment |
|---|---|---|---|
| `LEAN-LLM-OPT` | gpt-4.1, gpt-oss-20b | 已有 notebook | 主结果 |
| `Abl-RAGOnly` | gpt-4.1 | 已有 notebook，需重跑（分类模型变了） | 3.2 |
| `Abl-FewShotOnly` | gpt-4.1 | 已有 notebook，需重跑 | 3.2 |
| `Base-SingleCall` | gpt-4.1, gpt-5.2, gemini-3-pro | 已有 notebook，需补模型 | 4.1 的对比对象 |
| `Abl-NoWorkflow` | gpt-4.1 | 待实现 | 3.2 |
| `Abl-NoDataTools` | gpt-4.1 | 待实现 | 3.2 |
| `Abl-OracleType` | gpt-4.1 | 待实现 | 3.1 |
| `Abl-WrongRouting` | gpt-4.1 | 待实现 | 3.1 |
| `Base-OptiMUS` | 其原生配置 | 待实现 | 5 |
| `Base-OptiMUS-native` | 其原生配置 + NLP4LP | 待实现 | 5 |

每格建议 `n_repeats: 3`（`repeat_index=0/1/2`），报均值 ± 标准差。temperature=0 并不保证完全可复现，报单次结果审稿人有理由质疑。

---

## 10. 下一步：先做冒烟验证再全量跑

1. `export OPENAI_API_KEY=...`
2. 打开 `LEAN_LLM_OPT_4.1_Air_NRM.ipynb`，只跑 3–5 题。
3. 检查 `runs/<run_id>/instances.jsonl`：`n_llm_calls`、`prompt_tokens`、`by_stage` 都不为空，且 `token_source` 是 `provider` 而不是 `estimated`。
4. 对照原 notebook 的历史结果，确认 3–5 题的模型输出没有实质性退化（分类模型从 gpt-4 换成 gpt-4.1、agent 上限从无限变成 5，都可能改变输出）。
5. 确认无误后再全量跑，然后 `python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv`。

第 3 步如果 `token_source` 显示 `estimated`，说明这条链路的 usage 没传出来——OpenAI 走流式输出需要 `stream_usage=True`；Ollama 版本过旧也会缺 `usage_metadata`。这种情况要么升级依赖重跑，要么在论文里注明 token 数是估算值（汇总表有 `est_tokens_%` 列可以查）。
