# 实验运行指南

这套代码在原有 pipeline 外面加了一层**统一配置 + 运行计量**：每跑一道题，自动记录用了多少 token、调了几次 LLM、几次工具、花了多少钱、多长时间，并按流程阶段拆开。跑完可以直接生成对比表。

建模逻辑没有改动，改的是"用什么模型"和"怎么记账"。

---

## 一、跑通

### 1. 环境

```bash
conda create -n leanopt python=3.11 -y
conda activate leanopt
pip install -r requirements.txt
```

### 2. 填 API key

```bash
python set_key.py OPENAI_API_KEY
```

粘贴时**输入是隐藏的**（跟输密码一样，屏幕上不显示），回车即可。key 存在 `.env` 里，已被 `.gitignore` 排除，不会进代码、不会进日志。

查看当前配置（值自动打码）：

```bash
python set_key.py --show
```

### 3. 体检（不花钱，强烈建议）

```bash
python preflight.py
```

检查 Python 版本、依赖是否装齐且版本对得上、配置能否加载、数据文件在不在、
notebook 语法是否完好、key 有没有就位、Gurobi / Ollama 能否用、`runs/` 可不可写。
**全程不调 API。** 有 `FAIL` 就先修，别急着开跑。

`WARN` 不影响 —— 它标的是"只有某些 profile 才需要"的东西（比如你不跑
gpt-oss-20b，Ollama 连不上是正常的）。

### 4. 跑代码

打开 `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb`，**Restart Kernel**，从头运行到定义函数的最后一个 cell（cell 26）。

⚠️ **VS Code 用户注意**：如果有人更新了 `.ipynb` 或 `.py`，必须**关掉标签页重新打开 + 重启 kernel**，否则用的还是内存里的旧版本。这是最常见的"改了没生效"原因。

---

## 二、跑实验

### 先小规模试

```python
test = pd.read_csv('Test_Dataset/Large-scale-or/Large-scale-or-101.csv')
smoke = test.loc[[15, 45, 81]]        # NRM / RA / Others 各一道

output_model, output_code, classification = run_test(smoke, classification_agent)
```

3 道题约 $0.1、1 分钟。

⚠️ **不能用 `reset_index(drop=True)`**——instance_id 必须全局唯一，否则不同批次的结果会互相覆盖。

### 全量 101 题

```python
import leanopt_exp as lx

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

约 25–40 分钟，实测成本 **$3 左右**。

**中断了直接重跑这一格**，已完成的题会自动跳过，不会重复花钱。

### 单次调用 

```python
!{sys.executable} run_baseline.py --dataset Large-Scale-OR --profile gpt-4.1
```

先看开销不调 API：加 `--dry-run`。只跑指定题目：`--rows 15,45,81`。

> `!{sys.executable}` 保证用的是当前 kernel 的 Python。直接写 `python` 很容易跑到 conda base 上，报一堆 import 错误。

---

## 三、看结果

每次运行产生一个目录 `runs/<run_id>/`，`run_id` 形如：

```
LEAN-LLM-OPT__gpt-4.1__Large-Scale-OR__r0__20260731-142530__5f4c4fc93b8a
    方法        模型      数据集    重复次序   时间戳      配置哈希
```

| 文件 | 内容 |
|---|---|
| `config.json` | 完整配置 + 环境清单（包版本、Gurobi 版本、主机名） |
| `instances.jsonl` | **每题一行**：token、调用数、成本、延迟、分阶段拆分、完整 prompt |
| `calls.jsonl` | 每次调用一行，画图用 |
| `summary.json` | 本次运行汇总 |

**每题落盘**，跑到一半崩了不会丢数据。

看单次运行的消耗：

```python
import json
rows = [json.loads(l) for l in open(LOG.dir / "instances.jsonl")]
for x in rows:
    s = x['summary']
    print(f"[{x['instance_id']}] {x['status']:7s} LLM={s['n_llm_calls']:2d} "
          f"tool={s['n_tool_calls']:2d} in={s['prompt_tokens']:6d} "
          f"out={s['completion_tokens']:5d} ${s['cost_usd']:.4f} {s['wall_s']:.0f}s")
    for st, v in s['by_stage'].items():
        print(f"      {st:16s} LLM={v['calls']:2d} tool={v['tool_calls']:2d} "
              f"in={v['prompt_tokens']:6d} ${v['cost_usd']:.4f}")
```

输出长这样：

```
[15] ok      LLM= 7 tool= 2 in= 15700 out= 2420 $0.0379 18s
      classification   LLM= 3 tool= 1 in=  6535 $0.0125
      modeling         LLM= 3 tool= 1 in=  7679 $0.0203
      codegen          LLM= 1 tool= 0 in=  1486 $0.0052
```

---

## 四、生成论文用的表

跑完实验后，两步：

```bash
# 1. 执行生成的 gurobipy 代码，跟真值目标值比对
python score_runs.py runs/ -o gold_labels.csv --time-limit 60

# 2. 汇总出表
python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv
```

产出：

| 文件 | 内容 | 用途 |
|---|---|---|
| `tables/cost_table.csv` / `.tex` | 每个方法/模型的准确率、LLM 调用数、tool 调用数、token、成本、延迟 p95、失败率 | 主对比表，`.tex` 可直接 `\input` |
| `tables/stage_breakdown.csv` | 上述指标按 classification / data_retrieval / modeling / codegen 拆分 | 分析成本结构 |
| `tables/classification.csv` + `confusion__*.csv` | 分类准确率、混淆矩阵、分类正确 vs 错误条件下的建模准确率 | 分类依赖性分析 |
| `scored_all.csv` | 每题的 `n_vars` / `n_constrs` / `n_nonzeros` | 模型规模统计 |

`score_runs.py` 在**独立子进程**里执行生成的代码，死循环或崩溃不会拖垮评分器。

> `gold_labels.csv` 里的 `correct_formulation` 一列**故意留空**——它需要人工判定。最优值匹配会高估准确率：我们实测到过分类错误但目标值正确的案例。

### ⚠️ 出正式表格时，`score_runs.py` 不要带 `--run`

`-o gold_labels.csv` 是**整个覆盖**，不是追加。带了 `--run` 过滤器就只会写那一个 run 的结果，之前算好的全部被冲掉。

```bash
python score_runs.py runs/ --run Air-NRM -o gold_labels.csv    # 只调试某个 run 时用
python score_runs.py runs/ -o gold_labels.csv                  # 出表格用这个
```

没被评到的 run，表里准确率显示 `NaN`（诚实的"未测量"），不会顶着别人的分数——但样本量也就跟着少了。

### 两类"真值"不能混

| 类型 | 例子 | 属于 | 连接键 |
|---|---|---|---|
| 题目属性 | `gold_type`、`gold_objective`、`correct_formulation` | 题目本身，所有方法共享 | `(dataset, instance_id)` |
| 单次结果 | `correct_optimal`、`objective`、`n_vars` | **某一次运行**，各方法各不相同 | `(run_id, instance_id)` |

`aggregate_runs.py` 的 `correct_optimal` 一律从 `scored_all.csv` 按 `(run_id, instance_id)` 取。

**踩过的坑（2026-07-31）**：早期版本把 `correct_optimal` 也按 `(dataset, instance_id)` 连接，结果**最后评分的那个 run 的成绩会覆盖所有方法**。当时 Base-SingleCall 实测 3/3，表里却显示 66.7%——那是 LEAN-LLM-OPT 在同样三道题上的 2/3。

如果 `gold_labels.csv` 里出现了 per-run 的列却没有 `run_id`，`aggregate_runs.py` 会丢弃它们并打印警告，不会静默地算错。

### `run_notebook.py`：不开编辑器，直接跑几道题

```bash
python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb --list      # 只看计划，不花钱
python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb \
    --data Test_Dataset/Air_NRM/small_scale/query_CA.csv --n 2
python run_notebook.py LEAN_LLM_OPT_4.1_Large-scale-or.ipynb \
    --data Test_Dataset/Large-scale-or/Large-scale-or-101.csv --rows 15,45,81
```

它执行 notebook 的准备 cell，在**第一个"跑全量测试集"的 cell 之前停住**，然后自己调批处理函数、只喂你指定的那几行。同一份代码、同一个配置、同一套日志——省掉的只是那格会花十几倍钱的调用。

跑之前先 `--list` 看一眼切分点对不对（比如 `skipping : cell 17 onwards`）。切分是启发式的：找第一个**调用**（而非定义）`Batch_Process_Queries` / `run_test` 的 cell。



---

## 五、所有配置在一个文件里

`exp_config.yaml` 是唯一配置源，包括审稿人要求披露的全部内容：模型版本、解码参数、检索 k 值（22 个调用点逐一列出）、agent 迭代上限、重试策略、求解器时限、价格表。

配置文件的 sha256 会写进 `run_id`，任何一次运行都能反查到当时的确切设置。

几个常用项：

```yaml
common:
  budget_usd_per_run: 10.0      # 单次运行的硬性支出上限，超了自动停
  agent_max_iterations: 5       # agent 迭代上限，防止无限循环烧 token
  sleep_between_instances_s: 0  # 题目之间的等待
  log_prompts: true             # 是否记录完整 prompt
```

---

## 六、成本安全

这套代码有三层保护：

**预算上限。** `budget_usd_per_run: 10.0`，每题结束后检查累计花费，超了抛 `BudgetExceeded` 停下。全量 101 题实测 $3，$10 留了余量。

**agent 迭代上限。** `max_iterations: 5` + `max_execution_time: 300`。原代码没有上限，ReAct 解析失败会不停重试、prompt 逐轮膨胀，单题可能烧掉正常成本的几十倍。

**逐题记账。** 出现异常账单可以定位到具体是哪个 run、哪道题、哪个阶段。

另外提供一个静态审计脚本：

```bash
python audit_repo.py            # 扫描硬编码 key、可达模型名、失控风险
```

**养成习惯**：全量跑之前先用 3 道题实测单题成本，乘以题数，心里有数再开跑。

---

## 七、常见问题

**改了代码没生效** → VS Code 里关掉 notebook 标签页（选 Don't Save）→ 重新打开 → Restart Kernel。改 `.py` 只需重启 kernel，改 `.ipynb` 必须关掉重开。

**`ModuleNotFoundError`** → kernel 选错了。在 notebook 里 `import sys; print(sys.executable)` 看用的哪个 Python，然后 `%pip install -r requirements.txt`（带百分号，保证装进当前 kernel），装完重启 kernel。

**终端跑脚本报一堆 import 错误** → 同上，`python` 指向了 conda base。在 notebook 里用 `!{sys.executable} script.py` 最省事。

**`429 insufficient_quota`** → 不是限速，是账户余额不足。重试无用，需要充值或检查 project 的 budget limit。

**`credentials ready` 显示的 key 不对** → `.env` 改了但 kernel 里还是旧值。重启 kernel 即可（配置 cell 用的是 `load_dotenv(override=True)`，会以 `.env` 为准）。

**`token=ESTIMATED` 而不是 `provider`** → provider 没返回 usage，token 数是 tiktoken 估算的。论文里需要注明。遇到请反馈。

**找不到 `.env`** → 点开头的文件在 Finder 里默认隐藏，按 `Cmd + Shift + .` 显示。VS Code 侧边栏默认可见。或者用 `python set_key.py --show` 查看。

---

## 八、文件说明

| 文件 | 作用 |
|---|---|
| `exp_config.yaml` | **唯一配置源** |
| `leanopt_exp.py` | 配置加载、模型工厂、计量回调、运行日志 |
| `run_baseline.py` | 单次调用 baseline |
| `score_runs.py` | 执行生成代码、比对真值、产出 gold_labels |
| `aggregate_runs.py` | 汇总出表 |
| `switch_profile.py` | 切换模型 |
| `set_key.py` | 写入 API key |
| `preflight.py` | **开跑前体检**：依赖 / 配置 / 数据 / key / 求解器，不调 API |
| `run_notebook.py` | 终端里跑 notebook 的几道题，自动停在跑全量那格之前 |
| `link_air_nrm_labels.py` | 把 Air-NRM 的真值接进 query 文件（可重跑，`--check` 只验证） |
| `test_smoke.py` | 离线自检（用桩函数验证计量链路） |
| `audit_repo.py` | 静态安全审计 |
| `apply_instrumentation_patch.py` | 从 `original_notebooks/` 重新生成根目录的 notebook（改造脚本，可重跑） |
| 根目录 7 个 `.ipynb` | **改造后的 notebook，跑这些** |
| `original_notebooks/` | 改造前的原版，保留作对照，**不要跑**（见该目录下 README） |

数据集目录：

| 目录 | 内容 |
|---|---|
| `Test_Dataset/Large-scale-or/` | 101 题主数据集 |
| `Test_Dataset/Small-scale/` | NL4OPT / IndustryOR / MAMO |
| `Test_Dataset/Air_NRM/small_scale/` | Air-NRM 小规模 case（3 机场），notebook 现在跑的就是这个 |
| `Test_Dataset/Air_NRM/large_scale/` | Air-NRM 大规模 case（新航直飞两城市场，90 OD / 268 航班），已注册未接入 |

两个 Air-NRM case 数据同源、schema 相同，见 `Test_Dataset/Air_NRM/README.md`。

Air-NRM small_scale 的真值在 `SBLP_CA_Label/CA_answer.csv` 和 `SBLP_NP_Flow_Label/np_obj.csv` 里，由 `link_air_nrm_labels.py` 接进 query 文件的 `Label-objective` 列（对齐经过验证，见脚本 docstring）。large_scale 尚未生成标准 `.lp`，所以确实没有真值。

`PATCH_GUIDE.md` 记录了改造的全部细节和发现的问题。
