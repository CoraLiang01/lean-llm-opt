# LEAN-LLM-OPT · gpt-oss-20B 运行包

把论文里 **LEAN (gpt-oss-20B)** 那一路在 Large-Scale-OR（101 题）上重跑一遍，出
mean ± sd、成本/延迟表和分类混淆矩阵，外加 LOTO 五折 + Mixture。GPT-4.1 那一路已经跑
完了，这个包是同口径的另一半。

三个任务，三个目录，互不干扰：

| 目录 | 跑什么 | 规模 | 预计时间 |
|---|---|---|---|
| `large_scale_or/` | Large-Scale-OR 101 题主实验 × 3 次重复 | 303 个 instance-run | 6–9 小时 |
| `loto/` | LOTO 五折 + Mixture 折 | 307 个 instance-run | 6–9 小时 |
| `air_nrm_ls/` | Air-NRM 大规模 15 题（112–421 变量，4 个舱位等级）× 3 次重复 | 45 个 instance-run | 2–4 小时 |

**合计约 15–22 GPU 小时**（单卡 A100/4090 量级；显存够并行能压到一半）。按 GPT-4.1 的
实测外推：Large-Scale-OR 单次 101 题 = 697 次 LLM 调用、1.94M prompt tokens；
Air-NRM 大规模单次 15 题 = 75 次调用、25 分钟（4.1 上），gpt-oss 还要额外吐 reasoning。

---

## 0. 三条命令

先 activate 一个 Python ≥3.10 的环境。**每个任务先冒烟、再正式跑。**

冒烟（各约 10 分钟，装依赖 + 拉模型 + 自检 + 跑两三题 + 自动判卷）：

```bash
bash setup_and_smoke.sh large_scale_or
```

```bash
bash setup_and_smoke.sh loto
```

```bash
bash setup_and_smoke.sh air_nrm_ls
```

把三次打印的结论和生成的 `smoke_*.tar.gz` 发回来，**确认之后再开正式跑**：

```bash
nohup python run_oss20b.py --task large_scale_or --repeats 3 --output-dir runs > large_scale_or.log 2>&1 &
```

```bash
nohup python run_oss20b.py --task loto --output-dir runs > loto.log 2>&1 &
```

```bash
nohup python run_oss20b.py --task air_nrm_ls --repeats 3 --output-dir runs > air_nrm_ls.log 2>&1 &
```

看进度 `tail -f large_scale_or.log`。三个任务可以并行,但共享同一个 Ollama,串行反而更
省时间,除非显存够开 `OLLAMA_NUM_PARALLEL`。

下面几节是这些命令每一步在做什么,出问题时对着看。

---

## 1. 环境

```bash
conda create -n leanopt python=3.11 -y && conda activate leanopt
python -m pip install -r requirements-server.txt
```

版本是钉死的，别放开。notebook 用的是 `langchain_classic.agents.initialize_agent`
这类 LangChain 1.x 拆包之后才有的入口，装成别的大版本会在 cell 0 就炸。

**模型**（两个都是本地的）：

```bash
ollama pull gpt-oss:20b        # 约 13 GB，语言模型
ollama pull nomic-embed-text   # 约 274 MB，检索用的 embedding
ollama serve                   # 若不在 11434 端口：export OLLAMA_BASE_URL=http://host:port
```

**不需要任何 API key**，整条 pipeline 跑在这台机器上，不出网也能跑（模型拉下来之后）。

> 论文口径备注：GPT-4.1 那一路的 embedding 用的是 OpenAI 的
> `text-embedding-3-small`，这一路用本地 `nomic-embed-text`，两者向量空间不同。也就是
> 说两路之间的差距是"换了模型**且**换了检索"，写表的时候这一路的检索组件要单独说明。
> 想让两路严格可比的话，把 `exp_config.yaml` 里 `gpt-oss-20b` profile 的三行
> `embedding_*` 改成 openai / text-embedding-3-small 并配 `OPENAI_API_KEY` 即可。

如果这台机器起不了 20B，包里带了兜底路线：`PROFILE=openrouter-gpt-oss-20b`（需要
`OPENROUTER_API_KEY`，embedding 仍走本地 nomic-embed-text，和主路线保持一致）。
vLLM / SGLang 也可以，改 `exp_config.yaml` 里那个 profile 的 `base_url` 和模型 slug
即可，代码不用动。

---

## 2. 自检（不调模型，不花钱）

`setup_and_smoke.sh` 会自动跑这一步。想单独跑就三个目录各来一次：

```bash
cd large_scale_or && python ../verify_package.py
cd ../loto        && python ../verify_package.py
cd ../air_nrm_ls  && python ../verify_package.py
```

检查依赖版本、两个 profile 能否加载、两条路线的检索口径是否一致、数据行数与 sha256、
**被引用的每个 CSV 是否一个不缺**（Large-Scale-OR 侧 209 个、Air-NRM 侧 9 个固定输入
外加 15 条查询的 `Label-objective`）、notebook 能否编译、LOTO 六个折的 prompt 密封
断言。`0 FAIL` 才往下走。

（`hardware_note not filled in yet` 那条 WARN 是正常的，跑完再填。）

---

## 3. 冒烟（约 10 分钟）

```bash
bash setup_and_smoke.sh large_scale_or   # 3 题：NRM / RA / Others 各一
bash setup_and_smoke.sh loto             # AP 折，5 题 × 1 seed
bash setup_and_smoke.sh air_nrm_ls       # 2 题：112 变量和 421 变量，最小和最大
```

### 什么样算通过

**判卷是自动的**——`check_smoke.py` 会在冒烟结束后直接打 PASS/WARN/FAIL，不用肉眼看
JSON。它盯的是这个 port 的两个静默失败模式（下表），以及"每题都吐出了 gurobipy 代码"
这类基本条件。

要自己再看一眼原始记录：

```bash
python - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/*/instances.jsonl")):
    print("==", f)
    for line in open(f):
        r = json.loads(line); s = r["summary"]
        print(f"  [{r['instance_id']}] {r['status']:7s} pred={r.get('pred_type')!s:22s} "
              f"in={s['prompt_tokens']:6d} out={s['completion_tokens']:5d} "
              f"src={s.get('token_source')} stages={list(s['by_stage'])}")
EOF
```

| 看什么 | 正常 | 不正常说明什么 |
|---|---|---|
| `pred_type` | 三题各不相同（NRM / RA / Others 都出现） | **全是 "Others with CSV" ⇒ reasoning 文本混进了 `content`**。分类器是严格相等匹配，混进去就全部落到兜底分支。查 `exp_config.yaml` 里 `reasoning: true` 有没有生效 |
| `prompt_tokens` | Large-Scale-OR 15k–25k；Air-NRM 大规模 25k–48k（查询里内嵌了整张票价/需求/attraction 表） | **掉到几千 ⇒ `num_ctx` 没生效**，Ollama 把 prompt 前半截默默丢了（检索到的例子和数据表正好在前半截）。查 profile 里的 `num_ctx: 65536` |
| `completion_tokens` | 明显小于 32768 | 贴着上限 ⇒ 输出被截断，生成的代码是半截的 |
| token 计数 | `n_estimated_token_calls` 为 0 | 非 0 ⇒ 服务端没返回用量，token 和成本列是 tiktoken 估算的，要在论文里标注 |
| `cost_usd` | 0（本地模型没有 API 计费） | 非 0 ⇒ 跑到了托管路线上，不是本地模型 |
| `by_stage` | classification / modeling / codegen 三段都有数 | 缺段 ⇒ 某个阶段没被计量 |
| LOTO 那边 | 有 `[LOTO] Assertion passed: 'Assignment Problem' ...` | 没有 ⇒ 折没封住，这一折的数字不能用 |
| Air-NRM 那边 | `pred_type` 是 `CA` | `None` ⇒ 分类没出结果，同样是 reasoning 混进 `content` 的征兆 |

**把冒烟产生的 `runs/` 目录先发回来**，我们看一眼再开正式跑——一次全量是两三个小时，
配置错了发现得越晚越亏。

> 如果第一次调用就报 `think` 不支持，是 Ollama 版本的问题：把两份
> `exp_config.yaml` 里 `gpt-oss-20b` profile 的 `reasoning: true` 改成
> `reasoning: null`，再跑一次冒烟，然后告诉我们——那种情况下 `content` 里会带
> reasoning，解析层要改，不能直接开跑。

---

## 4. 正式跑

```bash
nohup python run_oss20b.py --task large_scale_or --repeats 3 --output-dir runs > large_scale_or.log 2>&1 &
nohup python run_oss20b.py --task loto --output-dir runs > loto.log 2>&1 &
nohup python run_oss20b.py --task air_nrm_ls --repeats 3 --output-dir runs > air_nrm_ls.log 2>&1 &
```

看进度：`tail -f large_scale_or.log`。

- **中断了直接重跑同一条命令。** 主实验会接着原来那个 run 目录往下写，已经记成 ok 的
  实例直接跳过；LOTO 那边被打断的那个 seed 会从头再跑（每折 5–25 题，损失有限），
  已经跑完的 seed 不动。
- 脚本开跑前会把之前冒烟产生的 run 挪到 `<任务>_runs_smoke/`。不挪的话，冒烟那几题会
  和正式数据混在一起被聚合——LOTO 尤其严重，等于同一折在两个 seed 下各算了一次，那一折
  的 n 就变了。挪完会打一个 `runs/.full_started` 标记，所以重跑不会把正式数据也挪走。
- 显存够可以 `export OLLAMA_NUM_PARALLEL=2`（或 3）提高吞吐。
- LOTO 那边会按折改写 `exp_config.yaml` 的 `holdout_type`，结束时自动还原成 `null`，
  失败也会还原。
- 换服务路线：`--profile openrouter-gpt-oss-20b`；换输出目录：`--output-dir runs_v2`。
- `python run_oss20b.py --task X --dry-run` 只打印它将要执行的命令，不跑任何东西。

**Large-Scale-OR 有一题注定跑不动**：instance 97 的单次 prompt 约 157k tokens，而
gpt-oss-20b 的上下文上限是 131k——不是 `num_ctx` 设小了，是这道题超出了模型的架构上限
（GPT-4.1 有百万级上下文所以能吃下）。它会被截断或失败，这属于"输入超长"而不是"建模
错误"，我们在表里会单独说明，不用管它。

**为什么要重复 3 次**：temperature=0 不等于确定性。GPT-4.1 在这同一个 benchmark 上跑
七次，准确率在 73.3%–78.2% 之间摆，101 题里约 19 题会随机翻面。单次运行的数字不能进
表，所以报 mean ± sd。

**LOTO 每折的 seed 数是不一样的**（NRM/RA/FLP/Mixture 各 3，TP/AP 各 5）——TP 和 AP
题太少，一题翻面就是 11–20 个百分点。这套 seed 数和 GPT-4.1 那张表完全一致，改了两个
面板就没法横向比了，所以别动。

---

## 5. 跑完回传

1. 三个 `runs_<任务>_<时间>.tar.gz`——每个任务跑完时 `run_oss20b.py` 已经自动打好包了，
   在包的根目录下。
2. 三个日志：`large_scale_or.log` / `loto.log` / `air_nrm_ls.log`
3. **硬件那一行**：GPU 型号、驱动/CUDA 版本、`ollama --version`、量化格式
   （`ollama show gpt-oss:20b` 能看到）。这行会原样进 `exp_config.yaml` 的
   `hardware_note` 再进论文的实验设置，没有它这批数字的墙钟和成本就没法写。

评分（执行生成的 gurobipy 代码、跟真值比对）默认我们这边做，你不用管。这台机器上如果
正好有 Gurobi 许可证，也可以自己出一版：

```bash
cd large_scale_or
python score_runs.py runs/ -o gold_labels.csv --time-limit 60
python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv
```

---

## 附：包里有什么

```
run_oss20b.py             三个任务的统一入口（冒烟 / 正式跑 / 打包）
setup_and_smoke.sh        首次上机：装依赖 + 拉模型 + 自检 + 冒烟
verify_package.py         离线自检
check_smoke.py            冒烟自动判卷
requirements-server.txt   钉死版本的依赖
.env.example              环境变量说明（没有值）
large_scale_or/           主实验：notebook + leanopt_exp.py + exp_config.yaml
                          + run_notebook.py + 数据 + 评分脚本
loto/                     LOTO：封堵过泄漏通道的 notebook 与 leanopt_exp.py
                          + run_loto_fold.py + loto_report.py + 数据
air_nrm_ls/               Air-NRM 大规模：Air_NRM notebook + 大/小规模两套数据
```

三个目录各带一份数据拷贝，互相独立，不用符号链接——链接在压缩传输里断掉之后报的是
"文件找不到"，很难查。

`air_nrm_ls/` 用的 notebook 是 `LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb`，
和小规模那份**不是**同一个文件。它把 GPT-4.1 那边
（`Test_Dataset/Air_NRM/large_scale/LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb`）
的大规模分叉原样搬了过来：

- 五张表全部来自 `large_scale/`，包含小规模没有的 `flight_capacity.csv`（小规模每班
  都是 187 座，数字是写死的）；
- **四个舱位等级**（Eco_flexi / standard / value / lite）而不是两个，从 v1.csv 的列名
  读出来；
- OD 代码是多字符（'865'、'678'），按正则捕获而不是按字符位置取；
- 容量按航班走，`flight_capacity` 是字典不是常数 187，生成代码的模板也相应按航班分组；
- CSV 和 FAISS 索引只建一次（这里 flight.csv 有 1072 行，小规模只有 38 行）。

包里**只带 large_scale 数据**，自检里有一条常驻检查：任何会被执行的 cell 里都不允许出现
小规模路径。混用不会报错，只会静悄悄地回答另一个问题。

`loto/` 里的 notebook 叫 `..._LOTO.ipynb`，和主实验那份**不是**同一个文件：它在分类
候选列表、few-shot、路由表、测试集过滤四处做了留出处理。别把两份互相覆盖。
