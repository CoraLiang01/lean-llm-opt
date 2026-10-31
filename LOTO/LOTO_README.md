# LOTO (Leave-One-Type-Out) 实验说明

**负责人**: ZCC  
**对应审稿意见**: R1.M1 (Referee 1, Major Comment 1)  
**目标**: 证明系统的建模能力不是靠"查到同类参考答案"，而是真正学到了可迁移的建模技能

## 核心思路

每次把一个问题类型从系统里"完全抹掉"（删掉它的参考语料、从分类器候选列表中移除、摘掉它的专用 workflow），然后用这个类型的测试题去考系统。如果系统仍然能做对一部分题，说明能力是可迁移的，不是靠查表。

## 四个 Tasks

### Task 1: 封闭泄漏通道

**做什么**: 改代码，确保被留出的类型 t 的信息不会从任何通道泄露进 pipeline。

需要封闭的 4 个通道:

| 通道 | 文件位置 | 怎么封 |
|------|----------|--------|
| (a) Ref-Data 检索语料 | `leanopt_exp.py` 的 `load_refdata_docs()` | 按 `problem_type` 过滤掉类型 t 的所有条目 |
| (b) 分类器候选列表 | notebook cell 4 的 `prefix` 变量 | 从类型列表中去掉 t |
| (c) Few-shot demo | notebook cell 4 的 `few_shot_examples_csv` | 如果正例恰好是 t（如留出 NRM 时），换成别的类型的 demo |
| (d) Type-tailored workflow | notebook cell 23 的 `run_test` 的 `elif` 链 | 摘掉 t 的分支，让它自然落入 `else`（通用 workflow） |

最后加一个**断言**: 检查最终 prompt 文本里不包含被留出类型的名字，作为安全守门员。

**满足的审稿要求**: R1 Q1 的 "does not rule out structural overlap"、R2 的 dataset leakage 质疑

### Task 2: 跑 5 折 LOTO 实验

**做什么**: 对 5 个有专用 workflow 的类型各做一折，每折只跑该类型在 benchmark 里的测试题。

| 折 | 留出类型 | benchmark 题数 | seeds |
|----|----------|---------------|-------|
| F1 | NRM | 25 | 3 |
| F2 | RA | 22 | 3 |
| F3 | FLP/UFLP | 14 | 3 |
| F4 | TP | 9 | 5 |
| F5 | AP | 5 | 5 |

说明:
- Mixture 和 Others 本来就走通用 workflow，留出它们是退化实验，不做
- AP 只有 5 题，单题翻转 = 20 个百分点波动，只作定性观察，不进主结论
- TP/AP 题数少所以多跑几个 seed 来降低噪声
- 分类器自由选择剩余类型，不做任何强制路由（强制路由是 LK 的 E3 实验）

**满足的审稿要求**: R1 Q1 的 "leave-one-type-out evaluation"、AE Q2 的 "close resemblance" 质疑

### Task 3: 附带损伤检查 (Collateral Check)

**做什么**: 每折在非留出类型的 20 题固定子集上也跑一遍，确认移除语料没有拖累其他类型的表现。

**为什么要做**: 移除某个类型会让 Ref-Data 变小（比如移除 RA 会砍掉 25% 的检索语料）。如果其他类型的精度也跟着下降了，说明 LOTO 观察到的衰减部分来自"语料变少"而非"类型缺失"。审稿人一定会问这个问题。

**满足的审稿要求**: 预防性，区分"类型缺失效应"和"语料缩减效应"

### Task 4: 整理结果表

**做什么**: 产出两张表。

**表 1 — 主结果表** (每折的精度对比):

| 留出类型 | n | A0 Full (已有) | LOTO | Agnostic (已有) | Base LLM (已有) |
|----------|---|----------------|------|-----------------|-----------------|
| NRM | 25 | 72.0% | _% | 20.0% | 20.0% |
| RA | 22 | 86.4% | _% | 45.5% | 18.2% |
| FLP | 14 | 78.6% | _% | 7.1% | 21.4% |
| TP | 9 | 100.0% | _% | 44.4% | 22.2% |
| AP | 5 | 100.0% | _% | 80.0% | 0.0% |

- A0 Full、Agnostic、Base LLM 三列论文里已有数据，不需要重跑
- LOTO 列是我们要跑出来的核心结果
- 逐折报告，不做跨折汇总（n 差 5 倍，加权平均无意义）

**表 2 — 路由分布表**: 分类器把留出类型的题实际分到了哪些剩余类型。比如 NRM 被留出后，25 道 NRM 题里有多少被分到了 RA、多少被分到了 Others，等等。

**满足的审稿要求**: R1 Q1 的完整回应

## gpt-oss-20B 版本 (2026-08-20)

同一套折、同一套泄漏封堵，换基座模型再跑一遍，对应论文里的 LEAN (gpt-oss-20B)。

实际跑是在同学的服务器上，用 `oss20b_run/loto/`（`build_oss20b_package.py` 生成的
自包含包，notebook 在那里叫 `..._LOTO.ipynb`）。在本仓库里等价的命令是：

```bash
# 单折
python run_loto_fold.py --holdout "Assignment Problem" \
    --notebook LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb \
    --profile gpt-oss-20b

# 五折 + Mixture（与 GPT-4.1 同口径，跳过 Others 折）
python run_loto_fold.py --all --skip Others \
    --notebook LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb \
    --profile gpt-oss-20b
```

`gpt-oss-20b` 是本地 Ollama（模型和 embedding 都在本地，不需要 key）；
`openrouter-gpt-oss-20b` 是跑不动 20B 时的托管兜底。

`--notebook` / `--profile` 是这次新加的。`run_loto_fold.py` 原来写死了
notebook 名和两个 cell 序号（config cell 1、last cell 33）；两个 notebook 的
布局不一样（oss 的 config 在 cell 0、Large-Scale-OR 段到 cell 26 为止），所以
现在这两个序号是**扫出来的**：config cell = 第一个含 `lx.load_config(` 的
cell，边界 = 第一个提到 NL4OPT / IndustryOR / MAMO 的 cell 往前一格。

### oss 版的三处不同

| 通道 | GPT-4.1 notebook | gpt-oss notebook |
|------|------------------|------------------|
| (b) 候选列表 | cell 4，`lx.get_candidate_types(CFG)` 从 `candidate_types` 生成 | cell 6，`lx.filter_holdout_types(CFG, ALLOWED_CATS)` 过滤它自己那份更短的硬编码列表 |
| (c) Few-shot demo | cell 4，留出 NRM 时整段换成 RA demo | cell 6，分类 few-shot 是从 RefData 动态检索的，通道 (a) 已经封住；**但 system prompt 的第 3 条规则里硬编码了 "resource allocation"**，留出 RA 时改写成不点名任何类型的说法 |
| (d) 路由 | cell 23 | cell 24，同样的 `_FULL_ROUTES` + `get_loto_routing_table` |

两点值得单独记一笔：

1. **断言改成了按词匹配（`\b...\b`）而不是子串匹配。** 别名组里有 `RA`、`AP`
   这种两字母缩写，子串匹配会在 "operations-**ra**esearch" 这类词里误报，每个 RA
   折都会被自己的守门员拦下来。
2. **`named_holdout` 在 oss 上差点变成恒等于 0 的假指标。** oss 的
   `classify_problem()` 会把任何不在 `ALLOWED_CATS` 里的回答折叠成
   "Others with CSV" *再*返回，批处理循环因此永远看不到模型说出被留出的类型名。
   现在在折叠**之前**用 `lx.record_extra(pred_type_raw=...)` 记一份原始回答，
   `loto_report.py` 优先读它——这样 `named_holdout = 0` 才是证据，而不是同义反复。

## 文件说明

| 文件 | 来源 | 是否需要修改 |
|------|------|-------------|
| `LEAN_LLM_OPT_4.1_Large-scale-or.ipynb` | 原目录副本 | 是 (cell 4 + cell 23) |
| `LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb` | 原目录副本 | 是 (cell 6 + cell 24 + cell 26) |
| `leanopt_exp.py` | 原目录副本 | 是 (load_refdata_docs 加过滤、`filter_holdout_types`、`record_extra`) |
| `exp_config.yaml` | 原目录副本 | 是 (加 holdout_type 等配置键、gpt-oss 的两个 profile 与 `num_ctx` 等本地推理参数) |
| `aggregate_runs.py` | 原目录副本 | 可能 (加 LOTO 结果分组) |
| `score_runs.py` | 原目录副本 | 不改 |
| `compare_arms.py` | 原目录副本 | 可能 |
| `gold_full.csv` | 原目录副本 | 不改 |
| `Large_Scale_Or_Files/` | 原目录副本 | 不改 (RefData 过滤在代码侧做) |
| `Test_Dataset/` | 符号链接 → 原目录 | 不改 |

## 类型分布速查

| 类型 | Ref-Data 条数 | Benchmark 题数 | instance_id 范围 |
|------|-------------|---------------|-----------------|
| NRM | 5 | 25 | 9-33 |
| RA | 24 | 22 | 34-56 (不含49) |
| UFLP/FLP | 9 | 14 | 57-65, 75, 78, 87, 88, 98 |
| TP | 7 | 9 | 0-8 |
| AP | 8 | 5 | 66-69, 92 |
| Mixture | 24 | 18 | 散布 |
| Others | 16 | 8 | 散布 |

注意 NRM 的不对称: Ref-Data 只有 5 条参考却有 25 道测试题，且论文里精度达到 72%。这本身已是"非查表"的弱证据，LOTO 把它变成强证据。
