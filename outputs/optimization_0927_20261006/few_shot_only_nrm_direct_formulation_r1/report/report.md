# Few-shot Only：NRM 直接数值 Formulation 运行结果

完整重新运行 101 题。模型 `gpt-4.1-2025-04-14`；匹配容差 rel_tol=abs_tol=1e-4。分类缓存、数据、示例、求解和其他 route 实现保持一致。

| 指标 | 修改前 | 本轮 | 变化 |
|---|---:|---:|---:|
| Objective match | 92/101（91.09%） | 88/101（87.13%） | -4 题（-3.96 个百分点） |
| Solved | 96/101（95.05%） | 94/101（93.07%） | -2 题（-1.98 个百分点） |

相对已完成全量 97/101 Objective match、101/101 Solved，本轮分别少 9 题、7 题。

## 按真实类别统计

| 类别 | 题数 | 修改前 Objective match | 本轮 Objective match | 本轮 Solved |
|---|---:|---:|---:|---:|
| TP | 9 | 9 | 9 | 9 |
| NRM | 25 | 23 | 20 | 22 |
| RA | 22 | 22 | 22 | 22 |
| FLP | 14 | 14 | 14 | 14 |
| AP | 5 | 5 | 5 | 5 |
| Mixture | 18 | 13 | 12 | 15 |
| Others | 8 | 6 | 6 | 7 |

## 失败、超时与不匹配

失败共 7 题；成功求解但 Objective 不匹配共 6 题。没有额度耗尽错误；所有记录保留，没有选择单题重跑。

| 题号 | 真实类别 | Route | 失败类型 / 不匹配目标值 |
|---|---|---|---|
| OR-013 | NRM | NRM | BadRequestError |
| OR-015 | NRM | NRM | APITimeoutError |
| OR-028 | NRM | NRM | ModelOutputTruncated |
| OR-080 | Mixture | Others | KeyError |
| OR-083 | Mixture | Others | ValueError |
| OR-085 | Others | Others | ValueError |
| OR-097 | Mixture | Others | TypeError |
| OR-023 | NRM | NRM | 61648.0；参考 63628.0 |
| OR-030 | NRM | NRM | 33651.32；参考 34856.76 |
| OR-087 | Mixture | Others | 495705.4；参考 6243055.96 |
| OR-094 | Mixture | RA | 410.4；参考 0.6 |
| OR-100 | Others | RA | 930862.0；参考 1382200.722 |
| OR-101 | Mixture | RA | 38969.9203187251；参考 25820.15 |

## 逐题变化与范围

由不匹配变为匹配：OR-071, OR-090。
由匹配变为不匹配/失败：OR-023, OR-028, OR-030, OR-080, OR-094, OR-097。

本轮只修改 NRM 为数值 Formulation → 自包含代码，没有运行时数据注入；其他 route 也完整重新生成，其变化包含生成波动，不能把总分变化全部归因于 NRM。

OR-013 为完整 Observation 超出上下文；OR-028 为建模输出截断；OR-015 为 API 超时。具体错误与其他失败详见 unsuccessful_cases.csv 和逐题 run.log。

分类相对前轮改变 0 题；25 个分配到 NRM 的案例均使用 legacy 自包含代码路径，记录中的 NRM 运行时数据注入为 0。

冻结 notebook SHA-256：`dfefa2c5f35373da2a8fba11177b500e97183bf615b9892f4848585178e2650c`。

[完整逐题结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/report/all_cases.csv) · [全部不成功案例](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/report/unsuccessful_cases.csv) · [类别结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/few_shot_only_nrm_direct_formulation_r1/report/per_class.csv)
