# WL 结构相似度计算代码

本包整理当前审稿回复采用的方法，可从原始 LP 计算 15 个参考、101 个测试模型和 36 个变种的 WL 相似度，包含 leave-one-type-out（LOTO）列。原始 LP 只读，不修改 CSV、模型或实验标签。

## 文件

- `run_wl.py`：LP 读取、单变量约束归入变量边界、二部图构建、参考匹配、分类汇总和审计。
- `summarize_wl_group_loto.py`：对 Mixture 和 Others 联合排除全部 10 个参考后重新取最近匹配。
- `compare_wl_loto_rules.py`：并列普通 LOTO 与联合排除规则的类别汇总。
- `verify_reproduction.py`：与保存的原实验做数值回归核对（需要项目中的历史结果目录）。
- `wl_core.py`：共享标签字典、同步 WL 更新、累计标签计数和归一化内积。
- `input_manifest.csv`：152 个 LP 的相对路径、类别和实验输入 SHA-256；参考类别取自已有审计清单。
- `requirements.txt`：Python 依赖。需要可读取本地模型的 Gurobi 环境；不运行优化求解。
- `results/`：整理后从原始 LP 重新计算的结果；`reproduction_check.json` 记录与既有结果的核对。

## 运行

使用已经安装 Gurobi、NumPy、SciPy 的 Python 环境：

```bash
python3 /Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_code_package_20261015/run_wl.py \
  --root /Users/cora/Documents/GitHub/lean-llm-opt
```

如需要安装依赖：`python3 -m pip install -r requirements.txt`。Gurobi 的环境／授权配置由其官方安装流程完成。本项目现有环境已安装所需依赖。

默认计算 h=0,1,2,3，以 h=2 为主结果。`--out` 可指定新输出目录。路径清单相对 `--root` 解析，因此代码包可移动，不依赖原脚本所在位置。

默认检查输入哈希，避免数据变化后误以为仍在复现原表格。如果确实要对更新后的 LP 重算，可显式添加 `--allow-changed-inputs`，审计文件会注明未要求历史哈希匹配；这时结果属于新实验。

## 方法逐步说明

1. **读取模型与统一单变量限制。** 若一行只有一个非零系数，例如 `2x <= 6`，它被视为 `x <= 3` 的变量限制。负系数会反转不等号；单变量等式同时限制下界和上界；多个限制与原生 Bounds 求交。整数边界按可行整数格点取 ceil/floor。记录边界，但这些行不进入图。
2. **构建变量—约束二部图。** 保留所有变量，包括固定、孤立变量；其余每行约束一个节点。非零系数对应一条无向边。变量标签分别为 `V:B`、`V:I`、`V:C`；等式为 `R:E`，两种方向的不等式均为 `R:I`。不执行一般 presolve。
3. **WL 更新。** `l_0(v)` 为初始标签。第 i 轮邻居多重集合 `M_i(v)` 收集所有邻居的第 i−1 轮标签；排序后，与自身旧标签组成签名 `s_i(v)`。共享字典 `f` 把相同签名映射到同一标签。所有节点同步更新。
4. **统计与归一化。** 各轮每种标签的数量以 `(轮数, 标签)` 为坐标，累计第 0 至 h 轮形成特征向量。两个向量的内积除以其 L2 长度乘积即余弦相似度。不得仅用第 h 轮计数代替累计计数。
5. **参考匹配。** 对每个测试分别取同类别参考中的最大分数 `S_same`、全部参考中的最大分数 `S_all`、排除同类别全部参考后的最大分数 `S_LOTO`。
6. **汇总。** 类别表报告测试数、`S_same` 中位数／均值、`S_all` 中位数、`S_LOTO` 中位数。全部行直接汇总所有测试实例，不是对类别中位数求平均。当前基础入口的 `S_LOTO` 对每个类别移除自身参考；若采用 Mixture/Others 联合排除规则，运行项目根目录的 `summarize_wl_group_loto.py`。

Mixture 有 8 个参考，Others 有 2 个参考。基础 LOTO 两类分别排除各自参考；联合规则同时排除这 10 个参考，只用于 Mixture 和 Others 测试。其余五类各有一个参考。当前变种集仅含 AP、UFLP、Mixture、Others。

## 与旧实现的关系

`wl_core.py` 保留既有 `wl_bipartite.py` 的同一 WL 算法。新入口把此前分散的读取、图构建和 LOTO 汇总集中起来。当前方法只需要构建图：不会先导出新的 LP/MPS 再读回，因此无需历史转换脚本及其交叉导入，也不需要再运行求解等价性实验。单变量限制保存在审计记录中，原 LP 不变。图及结果是否与此前一致，以逐模型图统计和逐配对数值核对为准。

## 输出

- `category_summary_h2.csv`：主结果，两个测试集合分开汇总。
- `category_summary_all_depths.csv`：各深度汇总。
- `large_scale_101/pairwise_h2.csv`、`variants/pairwise_h2.csv`：测试×15参考矩阵。
- `large_scale_101/nearest_h2.csv`、`variants/nearest_h2.csv`：逐实例三种最近分数、匹配参考及 LOTO 排除列表。并列最大值按清单参考顺序选第一项，不影响分数。
- `audit.json`：输入哈希、图规模、被吸收的行、有效边界、版本与内置核对。
- `grouped_loto_results/`：Mixture/Others 联合排除的结果和与普通 LOTO 的对照表。

## 本次复现核对

已从清单指定的原始 LP 重新计算：152 个图统计记录、h=1/2/3 的 6,165 个测试—参考配对分数，以及 h=2 的 65 个分类汇总数值均与原结果完全一致，最大数值差为 0。输入文件哈希未改变。详情见 `results/reproduction_check.json`。

压缩包含代码、路径清单、中文说明与本次结果，不包含原始 LP 数据；运行时需通过 `--root` 指向已有项目数据。

## 解释边界

分数越高，所选编码的累计标签计数越接近；没有通用高低阈值。目标函数、系数值与正负、右端项、边界值和名称均不进入该图。LOTO 列是结构相似度，不是求解准确率。高分不保证模型等价，低分不证明业务新颖或框架鲁棒。当前 NRM 参考包含此前修改的共享容量行；AP 测试和参考还存在连续／二进制变量域差异，解读时应保留这些背景。

## 文献

- Shervashidze et al. (2011), *Weisfeiler–Lehman Graph Kernels*, JMLR 12:2539–2561：https://jmlr.org/papers/v12/shervashidze11a.html 。标签细化、累计计数与 subtree kernel 是方法依据。
- Gasse et al. (2019), *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*：https://arxiv.org/abs/1906.01629 。变量—约束二部图是表示依据；本包没有训练该论文的网络。

图的标签选择、Bounds 处理、主深度 h=2、余弦报告及参考池匹配属于本实验设置，不是 WL 原论文给优化模型规定的标准。
