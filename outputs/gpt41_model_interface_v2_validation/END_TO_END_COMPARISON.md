# GPT-4.1 三组 V2 抽样与历史结果配对

同一批六道题。历史全量来源为 `outputs/final_101_V2/automatic`；两个 LOTO 历史来源分别为 `examples_only_V1` 和 `examples_and_route_V1`。新结果按问题 ID 配对，目标值用相同的 `1e-4` 容差重算。

| 条件 | 历史求解 | 历史匹配 | V2 已记录 | V2 求解 | V2 匹配 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| full | 6/6 | 6/6 | 6/6 | 6 | 6 | complete |
| examples_only | 2/6 | 2/6 | 6/6 | 5 | 5 | complete |
| examples_and_route | 6/6 | 6/6 | 6/6 | 6 | 6 | complete |

完整的新成绩只在同一条件六道题全部保存后显示；尚未运行的题不会记为失败。

| 条件 | 题号 | 历史 route | 历史求解 | 历史匹配 | V2 route | V2 求解 | V2 匹配 | 模型对象恢复 |
|---|---|---|---|---|---|---|---|---|
| full | OR-002 | 待运行 | 是 | 是 | TP | 是 | 是 | False |
| full | OR-004 | 待运行 | 是 | 是 | TP | 是 | 是 | False |
| full | OR-018 | 待运行 | 是 | 是 | NRM | 是 | 是 | False |
| full | OR-036 | 待运行 | 是 | 是 | RA | 是 | 是 | False |
| full | OR-063 | 待运行 | 是 | 是 | FLP | 是 | 是 | False |
| full | OR-069 | 待运行 | 是 | 是 | AP | 是 | 是 | False |
| examples_only | OR-002 | TP | 是 | 是 | TP | 是 | 是 | False |
| examples_only | OR-004 | TP | 否 | 否 | TP | 是 | 是 | False |
| examples_only | OR-018 | NRM | 否 | 否 | NRM | 否 | 否 | False |
| examples_only | OR-036 | RA | 否 | 否 | RA | 是 | 是 | False |
| examples_only | OR-063 | FLP | 否 | 否 | FLP | 是 | 是 | False |
| examples_only | OR-069 | AP | 是 | 是 | AP | 是 | 是 | False |
| examples_and_route | OR-002 | Others | 是 | 是 | Others | 是 | 是 | False |
| examples_and_route | OR-004 | Others | 是 | 是 | Others | 是 | 是 | False |
| examples_and_route | OR-018 | RA | 是 | 是 | RA | 是 | 是 | False |
| examples_and_route | OR-036 | Others | 是 | 是 | Others | 是 | 是 | False |
| examples_and_route | OR-063 | Others | 是 | 是 | Others | 是 | 是 | False |
| examples_and_route | OR-069 | Others | 是 | 是 | Others | 是 | 是 | False |
