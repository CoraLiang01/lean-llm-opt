# WL 结构相似性分析：AE 回复说明、计算流程与 101 个模型的结果

更新日期：2026 年 10 月 6 日。本报告针对 **101 个 Large-Scale-OR 测试 LP 与当前 15 个参考 LP**，不包含此前的 36 个 variants。

英文审稿回复已直接更新在原来的 [AE_WL_response.tex](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_latest_20261003/AE_WL_response.tex)，新增回复内容为蓝色。本文解释其依据、实现、图示和结论边界。

## 1. 应怎样回答 AE 的问题

AE 的疑虑有两个层次：

1. **参考实例与测试实例是否存在结构上的接近或重叠？** 不能只用“类别名称不同”或自然语言语义相似度回答。需要直接比较导出的数学模型。
2. **新增约束、改变目标或增加业务要求后，方法还能否正确建模与求解？** 这需要实际运行方法并评估结果。结构相似性分数本身不能代替性能实验。

本次 WL 分析直接回答第一层，并明确第二层仍需性能证据。可以向 AE 报告“结构接近程度因类别而异，并且可以定位影响分数的因素”；**不能据此声称已经证明了新增约束或改变目标下的鲁棒性，也不能笼统声称参考库与测试集没有结构重叠。**

## 2. 为什么采用 WL：文献各自支持什么

| 文献 | 论文做了什么 | 对本分析的具体支持 | 不能从该论文推出的结论 |
| --- | --- | --- | --- |
| Shervashidze et al. (2011), *Weisfeiler–Lehman Graph Kernels* | 将迭代更新的节点标签计数构成图特征，利用内积比较图；本文使用其中的 WL subtree kernel | 提供标签细化、标签计数和核函数的算法基础 | 没有直接定义 LP 的节点标签，也没有给出优化问题“高/低相似度”的通用阈值 |
| Gasse et al. (2019), *Exact Combinatorial Optimization with Graph Convolutional Neural Networks* | 使用变量—约束二部图及其属性，学习 MILP 的分支决策 | 支持用变量与约束的连接关系表示 MILP | 没有验证本文采用的简化图或 WL 分数能够预测建模准确率 |
| Hosny and Reda (2024), *Automatic MILP Solver Configuration by Learning Problem Similarities* | 通过图表示和与求解成本相关的学习式相似性，寻找邻近实例并配置求解器 | 支持“用图表示比较 MILP 实例”这一研究思路 | 其度量是学习式方法，不能称为本文这个 WL 核的直接验证 |
| Chen et al. (2023), *On Representing Mixed-Integer Linear Programs by Graph Neural Networks* | 研究 MILP 图神经网络的表示能力，包括所研究网络不能区分某些可行与不可行模型的情形 | 提醒我们不能把图表示的一致等同于优化性质一致 | 并不意味着所有网络或所有图表示都有完全相同的限制 |
| Jahin et al. (2026), *A Weisfeiler–Leman Characterization of Global-Attention Graph Transformers for Mixed-Integer Linear Programs* | 在其输入表示和网络假设下，刻画一类 MILP 图 Transformer 与 1-WL 表达能力的关系 | 支持把 WL 当作结构表达能力的诊断工具，并解释其盲区 | 不是两个优化问题的相似度标定研究，也不证明本文分数与求解性能相关 |

原始来源：

- [Shervashidze et al.：JMLR 论文 PDF](https://jmlr.csail.mit.edu/papers/volume12/shervashidze11a/shervashidze11a.pdf)，重点为 Algorithm 1 和 Definition 4。
- [Gasse et al.：NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html)。
- [Hosny and Reda：期刊论文](https://doi.org/10.1007/s10479-023-05508-x)。期刊卷期年份为 2024，网上先行发表为 2023。
- [Chen et al.：论文记录与全文入口](https://arxiv.org/abs/2210.10759)，ICLR 2023。
- [Jahin et al.：arXiv 记录](https://arxiv.org/abs/2607.17570)。截至本次核对，记录注明被 TAG-DS 2026 接收；引用稿本仍以该 arXiv 版本为依据。

采用 WL 的实际好处是：不需要训练模型；不需要借用本方法的成功/失败标签；不依赖变量名称；每一步可以复算；新增显式约束或改变非零连接会影响节点邻域，因此可以描述局部连接模式的变化。

**文献支持链是“优化模型可以表示为图”＋“WL 可以比较带标签图”＋“这种表示存在可说明的表达能力限制”。它不是“已有文献证明本文的分数等于业务相似度”。**

## 3. 实际使用什么图：与用户截图的区别

当前实现采用变量—约束二部图，而不是截图中的“约束—系数—变量”图。后者增加了系数节点，还对系数分桶；这样会改变节点数、邻域和计算结果，因此不能用它来解释当前这张 101 实例的结果表。

对一个模型 \(P\)，以 \(A_{aj}\) 表示第 \(a\) 条显式线性约束中变量 \(x_j\) 的系数。构造

\[
G_P=(V,E,\ell),\qquad
V=\mathcal V_{\rm var}\,\dot\cup\,\mathcal V_{\rm row},\qquad
E=\{\{v_j,r_a\}:A_{aj}\ne0\}.
\]

通俗理解：一个变量画成一个圆，一条约束画成一个矩形。只要该变量出现在该约束中，就连一条线。

初始标签是：

| 节点 | 标签 | 含义 |
| --- | --- | --- |
| 变量 | V:B | 二进制变量 |
| 变量 | V:I | 一般整数变量 |
| 变量 | V:C | 连续变量 |
| 约束 | R:E | 等式约束 |
| 约束 | R:I | 不等式约束；\(\leq\) 与 \(\geq\) 不再区分 |

本次图保留变量类型、等式/不等式以及非零连接，**不纳入目标函数、系数数值及正负号、右端项、变量上下界和名称**。没有预处理，也没有先求解模型。

具体来说：

- 将 \(x_1+x_2\leq4\) 改成 \(x_1+0.05x_2\leq5\)，上述图不变。
- 将 \(x_1+x_2\leq4\) 改成 \(x_1\leq4\)，会删除一条边。
- 增加一条显式约束，会增加一个约束节点及对应的边。
- 将连续变量改为二进制变量，即使连接不变，标签也会变化。
- 写在 LP 的 Bounds 部分的上界不生成约束节点；如果它被导出为 Subject To 中的一条行，则生成节点。因此，模型导出的具体形式影响分数。

这是一种 **带变量域标签的模型非零支撑图诊断**。它不能覆盖 AE 所说的全部业务变化，尤其不能评估仅改变目标函数的情况。

![LP 到当前二部图的转换](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_lp_graphs.png)

图 1 的投稿矢量版本：[PDF](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_lp_graphs.pdf) · [SVG](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_lp_graphs.svg)。

## 4. 使用原论文符号说明 WL 计算

以下采用 Shervashidze et al. 的符号；针对 LP 的初始标签和最后的归一化是我们的应用设置。

| 符号 | 在本实现中的意思 |
| --- | --- |
| \(G=(V,E,\ell)\) | 输入的带标签图 |
| \(\mathcal N(v)\) | 节点 \(v\) 的邻居 |
| \(l_0(v)=\ell(v)\) | 节点的初始标签 |
| \(M_i(v)\) | 第 \(i\) 轮收集的邻居旧标签多重集 |
| \(s_i(v)\) | 自身旧标签与排序后邻居标签的组合 |
| \(f\) | 相同组合得到相同新标签，不同组合得到不同新标签 |
| \(l_i(v)\) | 第 \(i\) 轮得到的新标签 |
| \(G_i=(V,E,l_i)\) | 结构不变、标签更新后的图 |
| \(\Sigma_i=\{\sigma_{i1},\ldots,\sigma_{i|\Sigma_i|}\}\) | 第 \(i\) 轮所有图共享的标签集合 |
| \(c_i(G,\sigma)\) | 图 \(G\) 在第 \(i\) 轮具有标签 \(\sigma\) 的节点数 |
| \(\phi^{(h)}_{\mathrm{WLsubtree}}(G)\) | 第 0 至 \(h\) 轮全部标签计数拼接形成的向量 |
| \(k^{(h)}_{\mathrm{WLsubtree}}(G,G')\) | 两图计数向量的内积 |
| \(S_h(P,Q)\) | 本应用报告的余弦归一化核相似度 |

### 第一步：收集邻居标签

\[
M_i(v)=\{\!\{l_{i-1}(u):u\in\mathcal N(v)\}\!\}.
\]

“多重集”意味着重复数量要保留。邻居标签是 [C,C,I] 和 [C,I] 的两个节点不能得到相同签名。

### 第二步：组合、排序与压缩

将自身旧标签 \(l_{i-1}(v)\) 放在前面，再接上排序后的 \(M_i(v)\)，得到 \(s_i(v)\)，然后

\[
l_i(v)=f(s_i(v)).
\]

例如，原标签为不等式、连接两个连续变量的约束，其签名可写为

\[
(\texttt{R:I},(\texttt{V:C},\texttt{V:C})).
\]

连接三个连续变量的约束具有不同的签名。

所有图在同一轮共享压缩字典。某个签名在参考图中被编号为 \(\sigma_{12}\)，在测试图中出现时也必须编号为 \(\sigma_{12}\)。若每个图独立从 1 开始编号，两个无关签名可能被误认为相同。

程序用“旧标签＋已排序邻居标签”的元组作为字典键，避免字符串拼接歧义。标签编号仅是名称，编号大小没有相似度意义。

### 第三步：每轮计数，并拼接各轮结果

每一轮数每个标签出现多少次，得到 \(c_i(G,\sigma)\)，再将第 \(0,\ldots,h\) 轮的计数全部拼接：

\[
\phi^{(h)}_{\mathrm{WLsubtree}}(G)=
\big(c_0(G,\sigma_{01}),\ldots,c_0(G,\sigma_{0|\Sigma_0|}),
\ldots,c_h(G,\sigma_{h1}),\ldots,c_h(G,\sigma_{h|\Sigma_h|})\big).
\]

实现将“轮次”也加入特征键，因此不同轮次恰好使用同一数字编号时，不会错误地混在一起。

### 第四步：内积与归一化

原始 WL subtree kernel 为

\[
k^{(h)}_{\mathrm{WLsubtree}}(G,G')
=\sum_{i=0}^{h}\sum_{\sigma\in\Sigma_i}
c_i(G,\sigma)c_i(G',\sigma).
\]

直观理解：同一种标签在两个图中分别出现 \(a\) 次和 \(b\) 次，贡献就是 \(ab\)；将每一轮的共同贡献累加起来。

我们最后报告

\[
S_h(P,Q)=
\frac{k^{(h)}_{\mathrm{WLsubtree}}(G_P,G_Q)}
{\sqrt{k^{(h)}_{\mathrm{WLsubtree}}(G_P,G_P)
k^{(h)}_{\mathrm{WLsubtree}}(G_Q,G_Q)}}.
\]

分数在 \([0,1]\) 内，**越大表示这些归一化计数向量越接近**。0.60 不是“60% 的约束相同”，也不是“60% 的业务要求相同”。

本次主分析使用 \(h=2\)，即累积第 0、1、2 轮。**这是 1-WL 的两次细化，不是 2-WL。** 我们另外报告 \(h=1,3\) 的敏感性，不能因为更深一轮让分数变低，就选它作为“最不相似”的证据。

## 5. 手算示例：新增一条约束如何影响分数

图 1 中参考模型 A 是

\[
\begin{aligned}
\min\ &2x_1+x_2\\
\text{s.t. }&x_1+x_2\leq4,\\
&x_1-x_2\geq1,\\
&x_1,x_2\geq0.
\end{aligned}
\]

模型 B 改为目标 \(3x_1+x_2\)，并将第一条约束改为 \(x_1+0.05x_2\leq5\)、第二条约束右端改为 2。变量类型和非零连接相同，因此 **\(S_h(A,B)=1\)**。

模型 C 在 A 的基础上增加一条**显式约束行** \(x_1\leq3\)。在这个示例中，它作为行而不是 Bounds 字段表示，图增加一个约束节点和一条边。

![WL 逐轮更新与相似度计算](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_refinement.png)

图 2 的投稿矢量版本：[PDF](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_refinement.pdf) · [SVG](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_refinement.svg)。

### 第 0 轮：只比较初始标签数量

A 有 2 个连续变量、2 条不等式；C 有 2 个连续变量、3 条不等式。

\[
\text{共同贡献}=2\times2+2\times3=10,
\]
\[
\text{A 自身贡献}=2^2+2^2=8,\qquad
\text{C 自身贡献}=2^2+3^2=13.
\]

因此 \(S_0(A,C)=10/\sqrt{8\cdot13}=0.9806\)。此时只看到节点类型的数量，尚未比较连接邻域。

### 第 1 轮：看到新增行直接连接的变量

A 的两个变量都连接两条不等式。C 的 \(x_1\) 连接三条，\(x_2\) 仍连接两条，因此 \(x_1\) 的新标签与 A 中变量的标签不同。C 新增行只连接一个变量，与原来连接两个变量的行也不同。

这一轮的共同贡献为 6，A 自身贡献为 8，C 自身贡献为 7。累积前两轮后，

\[
S_1(A,C)=\frac{10+6}{\sqrt{(8+8)(13+7)}}=0.8944.
\]

### 第 2 轮：变化传播到原有约束

C 中原来的两条约束都连接了标签已经变化的 \(x_1\)，所以它们在这一轮也获得与 A 不同的标签。

这一轮共同贡献为 2；累积结果为

\[
S_2(A,C)=
\frac{10+6+2}{\sqrt{(8+8+8)(13+7+7)}}
=\frac{18}{\sqrt{24\cdot27}}=0.7071.
\]

| 轮次 \(i\) | 当轮共同贡献 | 当轮 A 自身贡献 | 当轮 C 自身贡献 | 累积相似度 \(S_i\) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 10 | 8 | 13 | 0.9806 |
| 1 | 6 | 8 | 7 | 0.8944 |
| 2 | 2 | 8 | 7 | 0.7071 |

这些值已经与项目当前的 WL 内核交叉核对。注意：这只是算法演示，**不是 LLM 优化流程在新增约束下的准确率实验**。一般模型中增加约束也不保证每个深度的归一化分数都单调下降。

## 6. 为什么相似度等于 1 也不等于模型完全相同

归一化后，特征向量成比例即可得到 1。除了数值变化被忽略之外，1-WL 本身也可能无法区分全局连接不同的图。

![WL 无法区分的全局连接示例](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_limit.png)

图 3 的投稿矢量版本：[PDF](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_limit.pdf) · [SVG](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_limit.svg)。

左图为一个 8 节点的交替环，右图为两个互不连接的 4 节点交替环。两图都有 4 个连续变量、4 条不等式，而且每个节点的度都是 2。每轮中，同类型节点都看到相同的邻居标签多重集，因此全部轮次的标签计数相同，\(S_h=1\)。但左图是一个连通分量，右图是两个；两图不可能同构。

因此：

- 分数低于 1，说明选定的归一化 WL 特征不完全一致；不意味着没有共享模板。
- 分数为 1，说明这组特征一致或成比例；不意味着 LP 等价，也不意味着一定同构。
- 结构描述不同，不自动推出 LLM 建模成功；结构描述相同，也不自动推出两个实例求解难度相同。

## 7. 101 个测试 LP 如何与参考库匹配

本次使用的实际路径为：

- 测试 LP：[generated_label_models](/Users/cora/Documents/GitHub/lean-llm-opt/generated_label_models)。
- 参考 CSV：[RAG_Examples_All.csv](/Users/cora/Documents/GitHub/lean-llm-opt/Large_Scale_Or_Files/RAG_Examples_All.csv)。
- 当前 15 个参考 LP：[Ref_Data_Large_Scale_LP](/Users/cora/Documents/GitHub/lean-llm-opt/Large_Scale_Or_Files/Ref_Data_Large_Scale_LP)。

注意：当前参考文件夹在 Large_Scale_Or_Files 下；本次没有误用其他旧版参考目录。

对类别为 \(g(P)\) 的测试模型 \(P\)，先分别与候选参考逐个比较，再取最大值：

\[
S_{\rm same}(P)=\max_{R\in\mathcal R_{g(P)}}S_2(P,R),\qquad
S_{\rm all}(P)=\max_{R\in\mathcal R}S_2(P,R).
\]

所以“同类别最近”指同类别候选中的**最高相似度**；“全参考库最近”指 15 个候选中的最高相似度。随后才对各类别的这些逐实例分数计算中位数和均值。

参考池数量为：NRM、RA、TP、AP、UFLP 各 1 个，Mixture 8 个，Others 2 个，共 15 个。**Mixture 的同类别结果仅在 8 个 Mixture 参考中取最大值，没有把 Others 混进来。** “全部”一行直接合并 101 个分数计算，不是对七行类别统计做简单平均。候选池越大，取最大值就有更多机会得到较高分，因此类别之间不能忽略参考数量差异。

## 8. 最新重算结果

| 类别 | 测试数 | 同类别最近相似度中位数 | 同类别均值 | 全参考库最近中位数 |
| --- | ---: | ---: | ---: | ---: |
| NRM | 25 | 0.6097 | 0.6097 | 0.6097 |
| RA | 22 | 0.6562 | 0.6391 | 0.6562 |
| TP | 9 | 0.6240 | 0.6314 | 0.6343 |
| AP | 5 | 0.0508 | 0.0551 | 0.4424 |
| UFLP | 14 | 0.5503 | 0.5557 | 0.5503 |
| Mixture | 18 | 0.4387 | 0.4699 | 0.5376 |
| Others | 8 | 0.2584 | 0.3025 | 0.3811 |
| **全部** | **101** | **0.6097** | **0.5339** | **0.6097** |

全部实例同类别分数的 IQR 为 [0.5077, 0.6558]，即第 25–75 百分位数，并非置信区间。共有 100 个实例的同类别分数严格小于 1；UFLP10 为 1，其带标签支撑图与 Ref02 同构。不能将“100 个分数小于 1”解释成“100 个模型都具有独立的新业务结构”。

全参考库匹配使 29 个测试实例的分数提高。全部实例的两个中位数相同，并不代表每个实例的两个分数都相同。

![101 个模型的分类分布](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_101_results.png)

图 4：淡蓝点为逐实例同类别分数，蓝菱形为同类别中位数，蓝线为 IQR，橙方块为全参考库中位数。NRM 的 25 个分数重叠；显示出来不是只有一个实例。投稿矢量版本：[PDF](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_101_results.pdf) · [SVG](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/figures/fig_wl_101_results.svg)。

### 应保留的三个解释

**其一，类别之间确实存在不同的特征接近程度，但没有通用的高低阈值。** RA 和 TP 的同类别中位数高于 Mixture 和 Others；这只是指定图表示下的比较，不是对业务创新性或鲁棒性的排序。不能把 0.6097 宣称为已由文献标定的“低相似度”。

**其二，AP 的极低分受到变量类型编码强烈影响。** 5 个测试 LP 使用连续变量，而参考 Ref01 使用二进制变量。仅移除变量域标签后，AP 的中位数升至约 0.5842，AP2 的无变量域区分支撑图与 Ref01 同构。因此 0.0508 不能单独作为新增业务要求或新建模模式的证据。这项检查也不裁定原始 AP 的连续变量建模是否合理，后者仍需逐个核对其数学条件。

**其三，NRM 的参考已根据此前观察修改，必须披露。** 当前 NRM 参考此前在检查测试相似度后增加了一个合成共享容量约束 \(\sum_j x_j\leq100\)。25 个 NRM 测试图由重复的相同变量—约束局部组件组成，彼此归一化相似度为 1；在内存中只删除参考新增行，测试—参考相似度便从 0.6097 恢复为 1。因此，当前 NRM 分数描述的是**修改后参考池**，不能当作测试集独立异质性的证据，也不能作为测试侧新增业务要求下的性能证据。本次没有再次修改任何参考或测试模型。

## 9. 深度敏感性与方法边界

| 累积深度 | 全部 101 个实例的同类别最近相似度中位数 |
| --- | ---: |
| \(h=1\) | 0.9007 |
| \(h=2\)，主分析 | 0.6097 |
| \(h=3\) | 0.4608 |

这说明浅层邻域仍有较大的重叠，细化后可进一步区分部分局部模式。三个分数来自不同特征表示，不应该选择最小的一个来制造“相似度不高”的印象。

当前 WL 表示忽略目标、系数和右端项，精确标签匹配又可能对变量域及邻域变化敏感。它适合作为可审计的补充结构分析，而不是完整的优化模型等价性检验、业务要求度量或泛化保证。

## 10. 如何补齐 AE 真正要求的鲁棒性证据

建议配对设计，先固定参考库和推理设置，再以独立验证的模型变体运行方法：

| 条件 | 变体设计 | 为什么这样设计 | 应验证什么 |
| --- | --- | --- | --- |
| 原始模型 | 原始问题描述、数据与标准模型 | 建立每个实例自身的基线 | 数学模型正确性、可行性、目标值准确性 |
| 新增约束 | 增加容量、覆盖、最低服务水平或耦合约束；避免仅添加冗余行 | 直接检验 AE 所说的额外限制 | 约束是否正确加入、是否漏掉原条件、配对准确率变化 |
| 改变目标 | 保持可行域、改变目标或增加有明确权重的新目标项 | 单独考察当前支撑图看不到的变化 | 目标表达及方向、解的目标值、准确率变化 |
| 综合业务变体 | 组合额外约束、目标变化或新的变量关系 | 检验真实业务变化中因素同时出现的情形 | 完整模型正确性、可行性、目标值及失败原因 |

独立标准模型应先校验，确保问题描述与变体数学模型对应；采用改变约束可行集合的非冗余变体时，需检查预期变化确实发生。结果可以按类别和条件报告成功数/总数、配对准确率变化及相应不确定性。若方法包含随机推理，应固定可比较的设置并报告重复运行。

**这些是建议实验设计，本次没有运行这些下游性能实验，也没有虚构其准确率。** 如果原论文已有独立变体实验，最终 AE 回复应另接该实验的实际结果；本次 WL 证据只能回答“与参考有多接近、哪些因素造成差异”。

## 11. 审稿回复的推荐结论

可以使用的结论是：

> 我们补充了对全部 101 个数学模型与当前参考库的 WL 结构诊断。该分析显示模型与参考的接近程度随类别变化，并揭示变量域、参考修改及邻域细化对分数的影响。它为参考重叠提供了透明、可复算的描述，但不能排除共享模板，也不能代替新增约束、改变目标和业务要求下的实际性能评估。因此我们将结论限定在所提供模型的结构描述层面。

**不建议使用**“WL 中位数小于 1，所以充分证明方法能够泛化到新业务要求”或“所有类别都与参考高度不同”等表述。AE 很容易从 AP 的变量域差异、NRM 的参考改动或浅层分数提出反例。

## 12. 可复算文件与完整性检查

- [分类汇总 CSV，含 \(h=1,2,3\)](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/category_summary.csv)。
- [101 个实例的 \(h=2\) 最近参考结果](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/nearest_h2.csv)。
- [参考 CSV 代码与 LP 的对应检查](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/reference_correspondence.json)。
- [本次文件哈希、汇总数值及 WL 不变性检查](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/response_verification.json)。
- [演示例子每轮的精确核与相似度](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/toy_kernel_results.json)。
- [演示例子的节点标签和签名](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/toy_round_counts.csv)。
- [5 篇引用文献的 BibTeX](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/WL_references.bib)。
- [所有示意图及结果图的生成代码](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/generate_wl_figures.py)。
- [WL 实现](/Users/cora/Documents/GitHub/lean-llm-opt/wl_bipartite.py)与[101 实例重算入口](/Users/cora/Documents/GitHub/lean-llm-opt/recompute_wl_101.py)。

本次重算的 116 个 LP 文件哈希与此前审计输入一致；15 个参考 CSV Code 构建出的模型全部通过 LP 对应检查。对应检查比较变量域、上下界、矩阵、约束方向、右端及目标，并在优化求解前停止。**它验证 Code 与 LP 对应，不等于独立证明每个自然语言问题的建模语义正确。**

从项目目录复算：

~~~bash
/usr/bin/python3 recompute_wl_101.py \
  --reference-dir Large_Scale_Or_Files/Ref_Data_Large_Scale_LP \
  --out outputs/wl_response_20261006

/usr/bin/python3 outputs/wl_response_20261006/generate_wl_figures.py
~~~

TeX 中的两幅算法示意图使用内嵌 TikZ，不依赖外部图片文件；四幅单独导出的 PDF/SVG 可用于论文正文或补充材料。原回复稿修改前的备份保留在 [AE_WL_response_before_20261006.tex](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_response_20261006/AE_WL_response_before_20261006.tex)。
