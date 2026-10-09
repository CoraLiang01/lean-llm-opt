# 全量 606 route 结果

101 题 × 六个 route。复用同一全量代码自动分类所选 route 的 101 个结果，包括四个目标不匹配；新增运行 505 个组合。没有按结果选择重跑。

| Route | Objective match | Solved | 失败 |
|---|---:|---:|---:|
| TP | 82/101 | 94/101 | 7 |
| NRM | 90/101 | 98/101 | 3 |
| RA | 90/101 | 98/101 | 3 |
| FLP | 85/101 | 98/101 | 3 |
| AP | 87/101 | 96/101 | 5 |
| Others | 93/101 | 98/101 | 3 |

## 按真实类别：Objective match

| 类别 | 题数 | TP | NRM | RA | FLP | AP | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP | 9 | 9/9 | 8/9 | 8/9 | 9/9 | 8/9 | 8/9 |
| NRM | 25 | 18/25 | 24/25 | 21/25 | 20/25 | 20/25 | 22/25 |
| RA | 22 | 18/22 | 20/22 | 22/22 | 18/22 | 20/22 | 21/22 |
| FLP | 14 | 13/14 | 13/14 | 14/14 | 14/14 | 14/14 | 13/14 |
| AP | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Mixture | 18 | 11/18 | 14/18 | 14/18 | 13/18 | 12/18 | 16/18 |
| Others | 8 | 8/8 | 6/8 | 6/8 | 6/8 | 8/8 | 8/8 |

## 按真实类别：Solved

| 类别 | 题数 | TP | NRM | RA | FLP | AP | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP | 9 | 9/9 | 8/9 | 9/9 | 9/9 | 9/9 | 8/9 |
| NRM | 25 | 20/25 | 25/25 | 23/25 | 23/25 | 23/25 | 24/25 |
| RA | 22 | 22/22 | 22/22 | 22/22 | 22/22 | 21/22 | 22/22 |
| FLP | 14 | 14/14 | 13/14 | 14/14 | 14/14 | 14/14 | 13/14 |
| AP | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Mixture | 18 | 16/18 | 17/18 | 18/18 | 18/18 | 16/18 | 18/18 |
| Others | 8 | 8/8 | 8/8 | 7/8 | 7/8 | 8/8 | 8/8 |

## 口径与路径

模型、示例、数据访问、Formulation、代码生成、求解与匹配容差均使用冻结全量代码；强制 route 组合跳过分类。真实类别只用于事后分组，参考答案只用于评分。

错误记录中的 route 元数据依据任务目录修正，原始 result.json 未改；results_before_route_metadata_normalization.csv 保留原汇总。分类准确率不适用于强制 route 实验。

完整逐题记录：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/forced/results.csv
日志/请求/输出：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/attempts/<route>/<题号>/
复用来源：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_review_v6_completed/attempts/<题号>/result.json
配置与来源：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/run_manifest.json
类别统计：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/report/per_route_per_class.csv
101×6 匹配矩阵：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/report/objective_match_matrix.csv
全部失败和不匹配：/Users/cora/Documents/GitHub/lean-llm-opt/outputs/optimization_0927_20261006/full_606_routes_v6_r1/report/unsuccessful_cases.csv

## 运行诊断与完整性

606 个唯一组合均已核对，每个 route 均为 101 题；源码与输入文件哈希未改变。24 题失败，55 题已求解但目标不匹配，共 79 条不成功记录。没有额度错误。

OR-084 / RA 在原定 1800 秒外部截止触发后记录为 TimeoutError（1814.317 秒含停止和清理）。其 22.85 可行值与参考一致，但没有达到 OPTIMAL，因此按统一口径不计 Solved 或 Objective match。该模型最初有 4800 个二元变量。

全量自动分类基线仍为 97/101 Objective match、101/101 Solved。该结果与固定使用某一个 route 的结果是两种不同实验口径。
