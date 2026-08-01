# 上手指南

从零到跑通实验。详细说明见 `README_EXPERIMENTS.md`。

---

## 一、装环境

```bash
git clone <仓库地址>
cd LeanOpt

conda create -n leanopt python=3.11 -y
conda activate leanopt
pip install -r requirements.txt
```

已经 clone 过的直接 `git pull`。

## 二、填 API key

```bash
python set_key.py OPENAI_API_KEY
```

粘贴时**输入是隐藏的**（跟输密码一样，屏幕上不显示），回车即可。key 存在 `.env` 里，已被 `.gitignore` 排除，不会进代码、不会进日志。

查看当前配置（值自动打码）：`python set_key.py --show`

## 三、体检——不花钱，务必先跑

```bash
python preflight.py
```

一次性检查 Python 版本、依赖是否装齐且版本对得上、配置能否加载、数据文件在不在、notebook 语法是否完好、key 有没有就位、Gurobi / Ollama 能否用、`runs/` 可不可写。**全程不调 API。**

- 有 `FAIL` 先修，别急着开跑
- `WARN` 一般不用管——它标的是"只有某些 profile 才需要"的东西。比如你不跑 gpt-oss-20b，Ollama 连不上是正常的

---

## 四、先跑 3 题（约 $0.13）

```bash
# 先看它会跑哪些 cell，不花钱
python run_notebook.py LEAN_LLM_OPT_4.1_Large-scale-or.ipynb --list

# 真跑
python run_notebook.py LEAN_LLM_OPT_4.1_Large-scale-or.ipynb \
    --data Test_Dataset/Large-scale-or/Large-scale-or-101.csv --rows 15,45,81
```

`run_notebook.py` 会执行 notebook 的准备 cell，**在"跑全量测试集"那一格之前停住**，然后只跑你指定的题。省得开编辑器、点十几格、还怕手滑碰到跑全量那格。

不想调 API 只想看 prompt 组装的话：

```bash
python run_baseline.py --dataset Large-Scale-OR --profile gpt-4.1 --rows 15,45,81 --dry-run
```

## 五、全量 101 题（约 $3–4，25–40 分钟）

**用 notebook 跑**，因为它有断点续跑。打开 `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb`，**Restart Kernel**，从头运行到定义函数的最后一格，然后新建一格：

```python
import leanopt_exp as lx
import pandas as pd

test = pd.read_csv('Test_Dataset/Large-scale-or/Large-scale-or-101.csv')

done = LOG.done_instance_ids()        # 中断后重跑会自动跳过已完成的
todo = test[~test.index.isin(done)]
print(f"共 {len(test)} 题，已完成 {len(done)}，本次跑 {len(todo)} 题")

try:
    output_model, output_code, classification = run_test(todo, classification_agent)
except lx.BudgetExceeded as e:
    print("预算保护触发:", e)

agg = LOG.close()
print(f"{agg['n_instances']} 题, ${agg['total_cost_usd']:.2f}, {agg['total_wall_s']/60:.0f} 分钟")
```

**中断了直接重跑这一格**，已完成的题会自动跳过，不会重复花钱。

> ⚠️ **切片时不要用 `reset_index(drop=True)`**——instance_id 必须全局唯一，否则不同批次的结果会互相覆盖。用 `.loc[]` 或 `.head()`。

## 六、出表

```bash
python score_runs.py runs/ -o gold_labels.csv --time-limit 60
python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv
```

第一步在**独立子进程**里执行模型生成的 gurobipy 代码、跟真值比对（死循环或崩溃不会拖垮评分器）；第二步汇总出表。

> ⚠️ **出正式表格时 `score_runs.py` 不要加 `--run`。** `-o` 是整个覆盖不是追加，加了过滤器只会写那一个 run 的结果，之前算好的全被冲掉。`--run` 只在调试单个 run 时用。

产出在 `tables/`：

| 文件 | 内容 |
|---|---|
| `cost_table.csv` / `.tex` | 主对比表：准确率、LLM 调用数、tool 调用数、token、成本、延迟 p95、失败率 |
| `stage_breakdown.csv` | 上述指标按 classification / data_retrieval / modeling / codegen 拆分 |
| `classification.csv` + `confusion__*.csv` | 分类准确率、混淆矩阵 |

`gold_labels.csv` 里的 `correct_formulation` 一列**故意留空**——它需要人工判定。只看最优值匹配会高估准确率：我们实测到过分类错误但目标值恰好正确的案例。

---

## 七、换模型

```bash
python switch_profile.py --list                    # 看每个 notebook 现在用什么
python switch_profile.py gpt-5.2                   # 全部切到 gpt-5.2
python switch_profile.py gpt-4.1 --only Air_NRM    # 只切 Air-NRM 相关的
```

它拒绝把本地模型（gpt-oss-20b）的 notebook 切成付费云模型，除非加 `--force`——防止手滑把本该跑 Ollama 的 notebook 全切到 GPT-5。

切完记得**关掉 notebook 标签页重新打开 + 重启 kernel**。

## 八、成本保护

三层，都写在 `exp_config.yaml`：

- **预算上限** `budget_usd_per_run: 10.0`，每题结束后检查累计花费，超了抛 `BudgetExceeded` 停下
- **agent 迭代上限** `max_iterations: 5` + `max_execution_time: 300`，防止 ReAct 解析失败时无限重试烧 token
- **逐题记账**，异常账单能定位到具体哪个 run、哪道题、哪个阶段

另有静态审计：`python audit_repo.py`（扫硬编码 key、可达模型名、失控风险）

**养成习惯**：全量跑之前先用 3 题实测单题成本，乘以题数，心里有数再开跑。

---

## 九、常见问题

**改了代码没生效** → VS Code 里关掉 notebook 标签页（选 Don't Save）→ 重新打开 → Restart Kernel。改 `.py` 只需重启 kernel，改 `.ipynb` 必须关掉重开。

**`ModuleNotFoundError`** → kernel 选错了。在 notebook 里 `import sys; print(sys.executable)` 看用的哪个 Python，然后 `%pip install -r requirements.txt`（带百分号，保证装进当前 kernel），装完重启 kernel。

**终端跑脚本报一堆 import 错误** → 同上，`python` 指向了 conda base。先 `conda activate leanopt`。

**`429 insufficient_quota`** → 不是限速，是账户余额不足。重试无用。

**找不到 `.env`** → 点开头的文件在 Finder 里默认隐藏，按 `Cmd + Shift + .` 显示。或用 `python set_key.py --show`。

---

## 十、目录说明

| 路径 | 内容 |
|---|---|
| 根目录 7 个 `.ipynb` | **改造后的 notebook，跑这些** |
| `original_notebooks/` | 改造前的原版，保留作对照，**不要跑** |
| `exp_config.yaml` | **唯一配置源**：模型版本、解码参数、检索 k、迭代上限、价格表 |
| `leanopt_exp.py` | 配置加载、模型工厂、计量回调、运行日志 |
| `runs/<run_id>/` | 每次运行的完整记录，见下 |
| `Test_Dataset/Large-scale-or/` | 101 题主数据集 |
| `Test_Dataset/Air_NRM/small_scale/` | Air-NRM 小规模 case（3 机场），notebook 现在跑的是这个 |
| `Test_Dataset/Air_NRM/large_scale/` | Air-NRM 大规模 case（新航直飞两城市场，90 OD / 268 航班） |
| `Test_Dataset/Small-scale/` | NL4OPT / IndustryOR / MAMO |

每次运行产生 `runs/<run_id>/`，`run_id` 形如：

```
LEAN-LLM-OPT__gpt-4.1__Large-Scale-OR__r0__20260731-142530__5f4c4fc93b8a
    方法        模型      数据集    重复次序   时间戳      配置哈希
```

里面有 `config.json`（完整配置 + 环境清单）、`instances.jsonl`（**每题一行**：token、调用数、成本、延迟、分阶段拆分、完整 prompt）、`calls.jsonl`、`summary.json`。**每题落盘**，跑到一半崩了不会丢数据。

配置文件的 sha256 写进 `run_id`，任何一次运行都能反查到当时的确切设置。

---

## 十一、最近的重要改动（2026-07-31）

如果你之前跑过、发现数字和以前不一样，原因在这里。这几处都是**评分与统计的口径修正**，不是模型本身的改动：

| 修了什么 | 之前的问题 |
|---|---|
| `score_runs.py` 执行生成代码时的工作目录 | 生成的代码有两种写数据路径的方式（仓库根相对路径 / 裸文件名），原来只支持一种，另一种一律 `FileNotFoundError` 被判为答错。现在两种都试 |
| `aggregate_runs.py` 重复记录去重 | 重跑 batch cell 会往同一个 run 追加记录，同一题被统计多次，样本量、均值、失败率都被污染 |
| `aggregate_runs.py` 准确率连接键 | `correct_optimal` 曾按 `(dataset, instance_id)` 连接，**最后评分的那个 run 的成绩会覆盖所有方法**。现在按 `(run_id, instance_id)` |
| `exp_config.yaml` 数据集列名映射 | 只认 `Label-objective`，NL4OPT / IndustryOR / MAMO 的真值列叫 `Label`，读不到就等于没有真值——判分时"没真值"和"答错"结果一样 |
| Air-NRM 真值接线 | 真值一直在 `SBLP_*_Label/` 里，但 query 文件没有对应列，所以 Air-NRM 一直无法判分。现由 `link_air_nrm_labels.py` 接上（对齐经过验证） |

**已验证可跑**：`preflight.py`、`test_smoke.py`、`run_baseline.py`（Large-Scale-OR / NL4OPT）、`score_runs.py`、`aggregate_runs.py`、`audit_repo.py`、`switch_profile.py`，以及 `LEAN_LLM_OPT_4.1_Air_NRM`、`Ablation_Study_Air_NRM_RAG_Only`、`LEAN_LLM_OPT_4.1_Large-scale-or` 三个 notebook 的小规模运行。

**尚未跑过**：`LEAN_LLM_OPT_gpt_oss_20b_*`（需要本地 Ollama）、`Benchmark_Base_Model_Small_Scale.ipynb`、Air-NRM 的 NP-Flow 分支、全量 101 题。
