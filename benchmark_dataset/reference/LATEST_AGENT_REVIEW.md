# 最新 agent 结果与针对性调整

本次结果：12 题中 8 题正确。AP1、AP2、AP3、RA1、RA2、RA4、RA13、RA14 正确；RA12、RA3、RA7、RA10 失败。分析的是用户指定 LEAN 工程的 results.csv 和对应 solve.py，未修改源工程。

| 题目 | 失败原因 | 本地诊断修正后 |
| --- | --- | ---: |
| RA12 | select_latest_records 调用漏传 cutoff_date；此外 usage 只筛门店，未筛日期、最新版本或去重，多张表用实体 ID 而不是 record_id 选择事件。 | 45493 |
| RA3 | quicksum([x[i] >= 1]) 把临时约束当成数值表达式。原来已有 minimum_lot*y <= x <= maximum_order*y，该错误附加约束多余。 | 71822 |
| RA7 | usage、capacity_ledger、benefit 等表直接累计旧版本、未来版本和重复传输。资源约束与类别最低数量冲突，模型无解；不是题目本身不可行。 | 63831 |
| RA10 | groupby.apply 的子表中没有 record_id，函数又读取这一列，触发 KeyError；本地环境已复现。修复读取后另有未授权商品的 bundle 奖励变量未约束，错误获得 431+352=783 美分。 | 98866 |

RA10 仅替换数据筛选函数时得到 99649，比参考值 98866 多 783；同时修复不可用商品的奖励变量和类别无条件下限后得到正确值。类别下限在原代码中被错误写成 minimum_quantity*category_active，也已在诊断副本中纠正。

## 数据集本次实际变化

- 仅修改 RA12、RA3、RA7、RA10 的 Query：统一筛选顺序、计划日期作用范围、记录编号保留、数量与选中状态的线性关系、组合奖励触发条件。
- RA7 的 benefit、usage、capacity_ledger、fx 共 9 个输入文件改为计划期有效快照，其余表继续保留历史事件与其他业务主体。仍是 25 个分片，所有文件路径不变。
- RA7 的有效逻辑表与修改前完全一致，全部 12 题 label 保持不变。
- 其余八题 Query 和输入文件均保持原样。原始 35 题 CSV 未改。
- copy.csv 与 benchmark_dataset/questions.csv 同步。旧版问题文件及 RA7 数据保存在 benchmark_archive/before_targeted_agent_tuning。

## 验证边界

全部 12 题已从实际输入 CSV 重新求解，核对约束、目标值与 LP 参考模型。RA7 原 agent 代码仅将路径指向新输入，即可得到 63831；未改其数据处理或建模代码。

另外三个失败实例使用本地诊断代码修正后得到正确值，说明参考问题可解且错误可定位，不代表只改 Query 就保证下一次模型生成的代码正确。未重新调用 Gemini 或用户 agent API。

本地诊断报告见 latest_agent_diagnostics/report.json；RA7 仅替换输入的复跑结果见同目录 RA7_input_only_replay.json。诊断代码仅用于排错，不属于模型应读取的输入。
