# 用Gemini 3.1 Pro的记录估算Gemini 3 Pro（3.0）

脚本为 `estimate_gemini_3pro.py`，只使用Python标准库，离线读取数值，不需要API key。

## 使用本机已有的五组运行

在项目根目录执行：

```bash
python3 estimate_gemini_3pro.py
```

原始运行日志只保留在本机，不随代码上传。若本机已有相应日志，默认读取这五组运行：101题SingleCall，以及CA-LS/NP-LS各自的SingleCall和Two-stage。输出到 `outputs/gemini_3pro_estimate/`：

- `estimates.csv`：每次运行的基准和thinking敏感性情景。
- `estimates.json`：同一组结果，加上假设、价格、源文件hash和日志核对数据。

每组默认计算thinking倍率0.75、1.0、1.25，并单独提供无缓存情景费用。倍率是人为设定的敏感性情景，没有从3.0/3.1配对实验拟合。

## 输入某个运行目录

```bash
python3 estimate_gemini_3pro.py \
  --run-dir runs/Base-SingleCall__gemini-3.1-pro__Large-Scale-OR__r0__20260918-231948__0f8513964a8c \
  --thinking-ratios 1 1.25 \
  --output-dir outputs/gemini_3pro_custom
```

`--run-dir`可重复传入多个目录。脚本从 `calls.jsonl`逐次计算，只接受Gemini 3.1 Pro的LLM记录；有 `summary.json`时核对token、调用次数和已记录费用。工具调用不算作额外LLM token，失败请求若已记录非零用量仍计入。日志中usage unavailable的请求数量会单独保存，其未记录消耗不能从0恢复。

若假设3.0没有缓存命中，添加 `--no-cache`。它只改变缓存输入收费，不减少总token。

## 手动输入已有总token和API cost

```bash
python3 estimate_gemini_3pro.py \
  --total-tokens 810416 \
  --api-cost 4.896942 \
  --output-dir outputs/gemini_3pro_manual
```

只有两个总数时，默认同用量、同费用代理估算，得到810,416 tokens、$4.896942。可显式设置两个独立比例：

```bash
python3 estimate_gemini_3pro.py \
  --total-tokens 810416 --api-cost 4.896942 \
  --token-ratio 1.1 --cost-ratio 1.2 \
  --output-dir outputs/gemini_3pro_manual_scenario
```

`token-ratio`和`cost-ratio`分别表示3.0/3.1的假设比例。两个总数无法识别输入、thinking、缓存和长上下文分布，因此该模式只做比例估算，不能计算thinking专属变化。

## 按组成估算的公式

记P为总输入、K为其中的缓存子集、O为可见输出、H为隐藏thinking。假设调用轨迹、输入、可见输出不变，令r为thinking倍率：

```text
T_3.0 = P + O + r H
C_3.0 = [(P-K) × p_in + K × p_cache + (O+r H) × p_out] / 1,000,000
```

缓存K已经属于P，不能再加到总token里。项目的Gemini `completion_tokens`仅含可见输出，thinking需单独相加。每次请求按它的完整输入P选价格档位，不能用整轮累计P判断。

脚本固定使用项目原计量口径：

| 每次请求的输入量 | 输入USD/百万 | 缓存输入USD/百万 | 输出及thinking USD/百万 |
|---|---:|---:|---:|
| ≤200,000 | 2.00 | 0.20 | 12.00 |
| >200,000 | 4.00 | 0.40 | 18.00 |

3.0标准$2/$12价格来自[Google发布说明](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-developers/)。缓存和长上下文价格在目标估算中沿用[3.1官方价格表](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-pro-preview)，明确视为代理假设。核对日期2026-10-31。已有五组运行的每次调用均未超过200k。

只估算推理token费用，不自动带入缓存存储、搜索/grounding或其他工具附加费。源记录的费用与按固定价格重算的费用分别保存，保留逐调用舍入差异。生成结果始终标为estimate，不改写3.1原始记录或3.0实测列。

也可在Python里调用：

```python
from estimate_gemini_3pro import Usage, estimate_call, estimate_totals

result = estimate_call(
    Usage(prompt_tokens=100_000, completion_tokens=1_000,
          reasoning_tokens=2_000, cached_prompt_tokens=50_000),
    thinking_ratio=1.25,
)
# total_tokens=103500，token_cost_usd=0.152

simple = estimate_totals(total_tokens=810416, api_cost=4.896942)
```

运行离线检查：

```bash
python3 -m unittest test_estimate_gemini_3pro
```
