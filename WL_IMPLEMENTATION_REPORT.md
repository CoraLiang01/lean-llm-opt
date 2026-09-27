# LP 变量—约束二部图的 WL 结构相似度：实现、结果与 MIPLIB 关系

**状态：**本报告对应当前工作区的 15 个参考 LP、101 个生成测试 LP，以及
`recompute_wl_101.py` 在 2026-09-26 生成的结果。它是一个结构诊断，不能作为
泛化能力、问题语义等价或求解性能的单独证据。

## 1. 为什么需要另做 WL 诊断

两个线性/混合整数规划模型可能具有相近的变量数量和系数统计，却有不同的变量—约束
连接方式；也可能具有相同的连接图，但目标、右端项、边界或业务语义不同。当前 WL
方法刻意只观察前一种“谁与哪条约束相连”的拓扑结构。它用于补充、而不是取代，
MIPLIB 风格的数值和宏观特征诊断。

本实现遵循 Weisfeiler--Lehman (WL) subtree kernel 的标签细化思想
[@Shervashidze2011]，并将 LP 写成变量—约束二部图。它没有声称是 MIPLIB 2017
[@Gleixner2021] 的 105 维特征复现。

## 2. 输入、版本与可复现性

### 2.1 测试模型

101 个测试 LP 位于：

```
generated_label_models/<category>/<instance>/<instance>.lp
```

它们由 `label generation code/*/*.py` 生成。重算脚本逐个检查文件存在，并将当前
SHA-256 写入输出的 `manifest.csv`。当前分析没有从 CSV 标签文本直接解析 LP，也没有
对模型进行 presolve。

### 2.2 参考模型

当前参考 LP 为 `Ref Data_Large_Scale_LP/1.lp` 至 `15.lp`，类别来自
`Large_Scale_Or_Files/RAG_Examples_All.csv` 的相同行。映射是 CSV 第 *i* 行对应
`i.lp`。这不是历史的 `instance_078.lp` 等 reference pool，也不能把本报告的数字
与使用该 reference pool 的旧表直接合并。

NRM 的当前参考为原有 Nike 鞋类单期案例，加一条合成但已披露的共享发货容量：

\[
  \sum_i x_i \leq 100.
\]

收入、库存、需求及原有逐商品库存/需求约束没有变化。这个修改使参考图出现一个
连接全部四个变量的约束节点；它不是网络收益管理的完整航空网络模型。

### 2.3 文件与运行入口

|文件|职责|
|---|---|
|`wl_bipartite.py`|独立的 LP 读取、二部图、WL 细化、核归一化及自检实现|
|`recompute_wl_101.py`|构建 15 个参考与 101 个测试的 manifest、重算并导出结果|
|`outputs/wl_101_current_bipartite_20260926/manifest.csv`|每个模型的路径、哈希、变量/约束/非零数|
|`pairwise_h{1,2,3}.csv`|每轮深度下的 101×15 相似度矩阵|
|`nearest_h{1,2,3}.csv`|每个测试模型的同类别最近参考和全库最近参考|
|`category_summary.csv`|按类别汇总的中位数与均值|
|`audit.json`|计算配置、类别计数及输入文件指纹|

从仓库根目录执行：

```sh
/usr/bin/python3 recompute_wl_101.py
```

依赖为 Gurobi Python API、NumPy、SciPy；该命令不调用 LLM、API 或优化求解，只读取
LP。Gurobi 仅用于解析 LP 与读取稀疏矩阵。

## 3. 图构造：本实现究竟比较什么

设一个线性模型有系数矩阵 \(A\in\mathbb R^{m\times n}\)。建立二部图
\(G=(V_C\cup V_R,E,\ell_0)\)：

\[
 V_C=\{v_1,\ldots,v_n\},\quad
 V_R=\{r_1,\ldots,r_m\},\quad
 (v_j,r_i)\in E\Longleftrightarrow A_{ij}\ne0.
\]

初始标签为：

|节点|标签|
|---|---|
|二进制变量|`V:B`|
|一般整数变量|`V:I`|
|连续变量|`V:C`|
|等式行|`R:E`|
|不等式行（\(\le\) 或 \(\ge\)）|`R:I`|

因此，当前 **typed** WL 使用：非零支撑模式、变量域、等式/不等式。它**不使用**：

- 目标函数和目标方向；
- 系数的值、符号或相对大小；
- RHS、变量上下界和变量/约束名称；
- 约束的业务含义、数据来源或自然语言描述；
- presolve 后的图。

这也是为什么本报告不应把 0.61 读成“61% 的业务内容相同”。它仅是这个明确图表示下的
归一化标签计数相似度。

## 4. WL subtree feature 的计算

第 0 轮使用 \(\ell_0\)。第 \(t+1\) 轮，对每个节点 \(v\) 形成签名：

\[
 \sigma_t(v)=\left(\ell_t(v),\operatorname{sort}\{\ell_t(u):u\in N(v)\}\right),
\]

然后给相同签名分配同一新标签：\(\ell_{t+1}(v)=\mathrm{code}(\sigma_t(v))\)。每一
轮所有图共享一个 `codebook`；这是可比较性的必要条件，不能对每张图独立编码。

令 \(c_t(G)\) 是第 \(t\) 轮每个标签出现次数的向量。累计 WL subtree 向量为：

\[
 \phi_h(G)=\big[c_0(G),c_1(G),\ldots,c_h(G)\big].
\]

代码计算原始 kernel \(K_h(G,H)=\phi_h(G)^\top\phi_h(H)\)，再做余弦归一化：

\[
 S_h(G,H)=\frac{K_h(G,H)}
 {\sqrt{K_h(G,G)K_h(H,H)}}.
\]

故 \(S_h\in[0,1]\)，数值越高表示在此表示下越接近。主报告使用 \(h=2\)，即累计
第 0、1、2 轮；\(h=1\) 与 \(h=3\) 是敏感性结果。余弦归一化是本项目的汇总选择，
不是 Shervashidze et al. 原文必须采用的唯一尺度。

对于测试实例 \(P\)、其类别 \(g(P)\) 的参考池 \(R_g\)，报告的“同类别最近参考”是：

\[
 S_{\mathrm{same}}(P)=\max_{R\in R_{g(P)}} S_2(P,R).
\]

“全参考库最近参考”将最大化范围改为所有 15 个参考。类别表再对同类测试实例的这些
逐题分数取中位数或均值。NRM、RA、TP、AP 和 UFLP 当前各只有一个同类参考，因而
“同类最近”就是与该一个参考的相似度；Mixture 有 8 个，Others 有 2 个。

## 5. 实现检查

`wl_bipartite.self_test()` 检查：

1. 重新排列节点编号不会改变相似度；
2. 复制相同但互不相连的图组件后，归一化相似度仍为 1；
3. 每个 kernel 矩阵对称，且对角线为 1。

重算脚本还检查：101 个测试实例唯一、116 个输入 LP 都存在、每个 kernel 对称且单位
对角线为 1。当前脚本会拒绝二次/一般/SOS 约束，而不会静默删去它们。

## 6. 当前 101 个实例的结果（typed、累计 h=2）

|类别|测试数|同类参考数|同类中位数|同类均值|全库最近中位数|
|---|---:|---:|---:|---:|---:|
|NRM|25|1|0.609661|0.609661|0.609661|
|RA|22|1|0.656169|0.639143|0.656169|
|TP|9|1|0.624035|0.631446|0.634281|
|AP|5|1|0.050796|0.055061|0.442390|
|UFLP|14|1|0.550288|0.555720|0.550288|
|Mixture|18|8|0.438719|0.469883|0.537556|
|Others|8|2|0.258361|0.302515|0.381121|
|全部 101 个实例|101|15（总数）|0.609661|0.533852|0.609661|

“全部”行是 101 个逐实例分数的汇总，**不是**类别中位数的平均。

AP 是一个必要的解释例外：当前 AP 参考 `1.lp` 有 9 个二进制变量和 6 条等式，五个
测试 AP 有连续变量和等式。因为 typed WL 明确把 `V:B` 与 `V:C` 分开，AP 的低同类
分数主要反映变量域编码不一致；它不能独自证明其业务约束或耦合结构特别不同。

NRM 的 25 个分数相同，是因为这些生成 LP 均由“每个商品连接库存行和需求行”的重复
局部模板组成。余弦归一化消除了纯复制数量的影响；参考模型新增的一条跨四个变量的
共享容量行才使相似度由原模型的 1 降至 0.609661。这不表示 25 个问题均只与参考有
60.9661% 相同。

## 7. 与 MIPLIB 2017 的关系：相同目标，不同测量层面

MIPLIB 2017 的作者先作 trivial presolving 和 canonicalization，再在最终 presolved
canonical representation 上定义 105 个实例特征，分为 11 个特征组 [@Gleixner2021]。
其中包括规模、变量类型比例、目标非零密度、目标系数、边界、矩阵非零统计、矩阵系数、
行动态性、约束 sides、约束分类及分解。原文还对目标向量和每行系数按各自最大绝对值
归一化，再依据特征组规则缩放，并在特征空间中比较实例。

|问题|MIPLIB 2017 特征空间|本项目 typed WL 二部图|
|---|---|---|
|直接文献来源|Gleixner et al. (2021)|Shervashidze et al. (2011)|
|对象|presolved canonical MIP 表示|当前导出的原始 LP|
|表示|105 维工程化数值/结构特征|WL 迭代的离散局部邻域计数|
|系数、RHS、边界|纳入多个特征组|完全忽略|
|变量域|按比例统计|按节点标签逐点编码|
|矩阵连接|行/列非零数、分解等摘要|显式变量—约束邻接|
|约束类别/分解|SCIP 分类、GCG 检测|没有|
|尺度处理|论文指定的行归一化与特征缩放|kernel 向量的余弦归一化|
|输出|特征距离/聚类与实例选择分析|\([0,1]\) 图核相似度|

所以，二者的共同点仅是都把抽象优化模型转为可比较的结构表征；二者不是同一算法，也
没有可换算的统一阈值。MIPLIB 2017 结果不能为 WL 的 0.61 提供“高/低”的官方阈值，
WL 的结果也不能声称复现了 MIPLIB 的 105 维实验。

最合理的报告方式是：若论文要回答“是否有数值尺度、变量边界、RHS 或系数动态性差异”，
报告 MIPLIB 风格特征；若要回答“变量和约束是否以相似方式连接”，报告 WL。若二者
结论不同，这恰恰说明模型在不同层面相似或不同，不能任选一个有利数字替代另一个。

## 8. 对 AE 问题可以与不可以说什么

这项分析可以诚实支持：当前 101 个 LP 与当前参考库在变量—约束拓扑上并不完全相同；
这种差异随类别变化；允许跨类别参考后，部分实例能找到更接近的图结构。它还可作为
“同类标签不保证相同的 LP 连接结构”的量化补充。

它不能支持：方法对新增业务约束、目标、数据扰动或未知建模原型必然鲁棒；参考与测试
在业务语义上有某个百分比相同；某类别比另一类别更现实或更难；或低 WL 分数本身导致
模型性能更好/更差。要回应这些主张，需要固定参考池，对独立验证的受控 formulation
variants 做成对性能实验，并报告准确率、求解成功率和错误类型。

当前 NRM 参考的共享容量是在查看这 25 个测试模型后设计的，因此此 NRM 结果尤其不能
作为独立外推证据。它适合作为公开披露的敏感性/结构诊断，而不应被写成无测试集调参
的泛化结论。

## 9. BibTeX

```bibtex
@article{Shervashidze2011,
  author  = {Shervashidze, Nino and Schweitzer, Pascal and van Leeuwen, Erik Jan
             and Mehlhorn, Kurt and Borgwardt, Karsten M.},
  title   = {Weisfeiler--Lehman Graph Kernels},
  journal = {Journal of Machine Learning Research},
  volume  = {12}, pages = {2539--2561}, year = {2011},
  url     = {https://jmlr.org/papers/v12/shervashidze11a.html}
}

@article{Gleixner2021,
  author  = {Gleixner, Ambros M. and Hendel, Gregor and Gamrath, Gerald
             and Achterberg, Tobias and Bastubbe, Michael and Berthold, Timo
             and Christophel, Philipp M. and Jarck, Kati and Koch, Thorsten
             and Linderoth, Jeff and L"{u}bke, Tobias and Mittelmann, Hans D.
             and Ozyurt, Derya and Ralphs, Ted K. and Salvagnin, Domenico
             and Shinano, Yuji},
  title   = {{MIPLIB} 2017: Data-Driven Compilation of the 6th Mixed-Integer
             Programming Library},
  journal = {Mathematical Programming Computation},
  volume  = {13}, pages = {443--490}, year = {2021},
  doi     = {10.1007/s12532-020-00194-3}
}
```

## References

- Shervashidze et al., *Weisfeiler--Lehman Graph Kernels*, JMLR 12 (2011),
  2539--2561: <https://jmlr.org/papers/v12/shervashidze11a.html>.
- Gleixner et al., *MIPLIB 2017: data-driven compilation of the 6th mixed-
  integer programming library*, Mathematical Programming Computation 13
  (2021), 443--490, DOI: <https://doi.org/10.1007/s12532-020-00194-3>.
