# A 与 C 的 WL 相似度：逐轮计算简明报告

WL 的做法是：**给节点贴标签 → 根据邻居更新标签 → 统计各轮标签数量 → 比较计数向量。** 本例累计到第 2 轮，得到相似度 **0.7071**。

**图编码约定：** 新增的 $x_1\leq3$ 作为约束节点，两个模型共同的非负 Bounds 不入图。这是教学例子的约定；将所有上下界统一处理后的结果另见第 7 节。

## 1. 模型如何变成图

模型 A 为：

$$
\begin{aligned}
\min\quad &2x_1+x_2\\
C_1:\quad &x_1+x_2\leq4,\\
C_2:\quad &x_1-x_2\geq1,\\
&x_1,x_2\geq0,\qquad x_1,x_2\text{ 为连续变量。}
\end{aligned}
$$

模型 C 保留 A 的全部内容，并新增 $C_3:x_1\leq3$。

每个变量、每条纳入分析的约束各生成一个节点。约束包含某变量的非零项，就连接两者；变量之间、约束之间不直接连边。本例不编码系数数值、正负号、目标和 RHS。

![模型 A、C 及其二部图](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_tutorial_20261006/figures/01_models_and_graphs.png)

A 有 4 个节点、4 条边；C 有 5 个节点、5 条边。下表合并了邻居相同的节点：

| 模型 | 节点 | 邻居 $\mathcal N(v)$ |
|---|---|---|
| A | $x_1,x_2$ | $\{C_1,C_2\}$ |
| A | $C_1,C_2$ | $\{x_1,x_2\}$ |
| C | $x_1$ | $\{C_1,C_2,C_3\}$ |
| C | $x_2$ | $\{C_1,C_2\}$ |
| C | $C_1,C_2$ | $\{x_1,x_2\}$ |
| C | $C_3$ | $\{x_1\}$ |

## 2. 更新公式与符号

图记为 $G=(V,E,\ell)$：$V$ 是节点集合，$E$ 是边集合，$\ell$ 是初始标签函数。A、C 各有自己的图；跨图使用同一套标签规则。

初始化为 $l_0(v)=\ell(v)$。第 $i\geq1$ 轮按三步更新：

$$
\begin{aligned}
M_i(v)&=\{\!\{l_{i-1}(u):u\in\mathcal N(v)\}\!\},\\
s_i(v)&=\operatorname{concat}\bigl(l_{i-1}(v),\operatorname{sort}(M_i(v))\bigr),\\
l_i(v)&=f(s_i(v)).
\end{aligned}
$$

| 符号 | 含义 |
|---|---|
| $v,u$ | 当前节点、它的某个邻居；都可以是变量或约束 |
| $i,\ i-1$ | 当前轮、上一轮 |
| $\mathcal N(v)$ | 直接连接 $v$ 的节点集合 |
| $l_i(v)$ | 节点 $v$ 在第 $i$ 轮的标签 |
| $M_i(v)$ | 邻居的上一轮标签组成的多重集，**保留重复项** |
| $\operatorname{sort}$ | 排序，消除邻居枚举顺序的影响 |
| $s_i(v)$ | 自己的旧标签与排序后的邻居旧标签组成的签名 |
| $f$ | 所有图共享的映射：相同签名给相同新标签，不同签名给不同新标签 |
| $\operatorname{concat}$ | 拼接；下文用 $\mid$ 分隔自己的旧标签和邻居列表 |

例如：

$$
M_1(x_1^A)=\{\!\{\mathrm R,\mathrm R\}\!\}
\;\longrightarrow\;
s_1(x_1^A)=\mathrm V\mid[\mathrm R,\mathrm R]
\;\longrightarrow\;
l_1(x_1^A)=a.
$$

上标 A 表示所属模型，不是幂次。字母 a 只是短代号，没有大小意义。

**同步更新：** 第 2 轮所有节点都读取第 1 轮标签，不能读取本轮刚算出的新标签。每轮只改标签，节点和边始终不变。

## 3. 第 0 轮：只贴类型标签

规定连续变量为 $\mathrm V$（实际标签 V:C），不等式为 $\mathrm R$（实际标签 R:I）。这里 $\mathrm V$ 是标签，与节点集合 $V$ 不同。

| 模型 | 节点的初始标签 | 按 $(\mathrm V,\mathrm R)$ 排列的计数 |
|---|---|---|
| A | $x_1,x_2$ 为 V；$C_1,C_2$ 为 R | $(2,2)$ |
| C | $x_1,x_2$ 为 V；$C_1,C_2,C_3$ 为 R | $(2,3)$ |

原论文在初始化时写 $M_0(v):=l_0(v)=\ell(v)$。**这个 $M_0$ 是初始化记号，不是邻居多重集。** 邻居公式从第 1 轮开始，不使用不存在的 $l_{-1}$，也不用计算 $s_0$。

## 4. 第 1 轮：读取邻居的初始标签

下面合并同样的节点，仍覆盖 A、C 的全部节点。表格最后一列同时规定了本轮共享映射 $f$。

| 模型与节点 $v$ | 自己的 $l_0(v)$ | $M_1(v)$ | 签名 $s_1(v)$ | 新标签 $l_1(v)$ |
|---|---|---|---|---|
| A 的 $x_1,x_2$；C 的 $x_2$ | V | $\{\!\{\mathrm R,\mathrm R\}\!\}$ | $\mathrm V\mid[\mathrm R,\mathrm R]$ | a |
| A、C 的 $C_1,C_2$ | R | $\{\!\{\mathrm V,\mathrm V\}\!\}$ | $\mathrm R\mid[\mathrm V,\mathrm V]$ | b |
| C 的 $x_1$ | V | $\{\!\{\mathrm R,\mathrm R,\mathrm R\}\!\}$ | $\mathrm V\mid[\mathrm R,\mathrm R,\mathrm R]$ | c |
| C 的 $C_3$ | R | $\{\!\{\mathrm V\}\!\}$ | $\mathrm R\mid[\mathrm V]$ | d |

例如，C 的 $x_1$ 多连接了一个约束，所以有三个 R，与 A 中变量的两个 R 不同，新标签为 c。

**两条原有约束暂时仍匹配：** 它们读的是变量的第 0 轮标签 V、V，不能提前读取 $x_1^C$ 本轮刚得到的 c。

按标签 $(a,b,c,d)$ 排列：

$$
\text{A 的计数}=(2,2,0,0),\qquad
\text{C 的计数}=(1,2,1,1).
$$

## 5. 第 2 轮：差异传播到原有约束

此时 A 的变量标签为 a、a，约束为 b、b；C 的变量为 c、a，约束为 b、b、d。

| 模型与节点 $v$ | 自己的 $l_1(v)$ | $M_2(v)$ | 签名 $s_2(v)$ | 新标签 $l_2(v)$ |
|---|---|---|---|---|
| A 的 $x_1,x_2$；C 的 $x_2$ | a | $\{\!\{b,b\}\!\}$ | $a\mid[b,b]$ | p |
| A 的 $C_1,C_2$ | b | $\{\!\{a,a\}\!\}$ | $b\mid[a,a]$ | q |
| C 的 $x_1$ | c | $\{\!\{b,b,d\}\!\}$ | $c\mid[b,b,d]$ | r |
| C 的 $C_1,C_2$ | b | $\{\!\{c,a\}\!\}$ | $b\mid[a,c]$ | s |
| C 的 $C_3$ | d | $\{\!\{c\}\!\}$ | $d\mid[c]$ | t |

**原有约束为什么变了？** 以 $C_1$ 为例：

$$
\begin{aligned}
M_2(C_1^A)&=\{\!\{l_1(x_1^A),l_1(x_2^A)\}\!\}
=\{\!\{a,a\}\!\},\\
s_2(C_1^A)&=b\mid[a,a],\qquad l_2(C_1^A)=q;\\[3pt]
M_2(C_1^C)&=\{\!\{l_1(x_1^C),l_1(x_2^C)\}\!\}
=\{\!\{c,a\}\!\},\\
s_2(C_1^C)&=b\mid[a,c],\qquad l_2(C_1^C)=s.
\end{aligned}
$$

两图的约束都连着两个变量，但这些变量的**上一轮标签不同**，因此约束的新标签不同。差异沿“新增约束 $C_3$ → 变量 $x_1$ → 原有约束 $C_1,C_2$”传播，没有新增约束—约束边。

C 的 $x_2$ 本轮仍匹配 A 的变量：它读取原有约束**上一轮的 b、b**，而不是本轮的新标签 s、s。标签 s 是代号，与签名函数 $s_i(v)$ 不同。

![第 2 轮差异如何传播](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_tutorial_20261006/figures/03_synchronous_propagation.png)

按 $(p,q,r,s,t)$ 排列：

$$
\text{A 的计数}=(2,2,0,0,0),\qquad
\text{C 的计数}=(1,0,1,2,1).
$$

![两图在三轮中的标签变化](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_tutorial_20261006/figures/02_rounds_same_graph.png)

## 6. 标签计数如何变成相似度

### 6.1 每轮贡献

$\Sigma_i$ 是第 $i$ 轮两图共享的标签集合；$\sigma$ 是其中一个标签；$c_i(G,\sigma)$ 是该标签在图 $G$ 中的节点数。

本报告用 $K_i$ 简称第 $i$ 轮的贡献：

$$
K_i(A,C)=\sum_{\sigma\in\Sigma_i}c_i(G_A,\sigma)c_i(G_C,\sigma).
$$

即**相同标签的数量相乘，再相加**。自身贡献则把各标签数量平方后相加。

| 轮次 | 两图共同贡献 $K_i(A,C)$ | A 自身贡献 $K_i(A,A)$ | C 自身贡献 $K_i(C,C)$ |
|---|---|---|---|
| 0 | $2\times2+2\times3=10$ | $2^2+2^2=8$ | $2^2+3^2=13$ |
| 1 | $2\times1+2\times2=6$ | $2^2+2^2=8$ | $1^2+2^2+1^2+1^2=7$ |
| 2 | $2\times1=2$ | $2^2+2^2=8$ | $1^2+1^2+2^2+1^2=7$ |

未写出的标签乘积均为 0。例如第 2 轮，A 有两个 p，C 有一个 p；其余标签没有跨图匹配，所以共同贡献只有 $2\times1=2$。

### 6.2 累计到第 $h$ 轮

$h=2$ 表示更新两次，并保留第 0、1、2 轮的全部计数，算法仍然是 1-WL。

$\phi^{(h)}_{\mathrm{WLsubtree}}(G)$ 是各轮计数拼成的特征向量。本例：

$$
\begin{aligned}
\phi^{(2)}_{\mathrm{WLsubtree}}(G_A)
&=(2,2\,;\;2,2,0,0\,;\;2,2,0,0,0),\\
\phi^{(2)}_{\mathrm{WLsubtree}}(G_C)
&=(2,3\,;\;1,2,1,1\,;\;1,0,1,2,1).
\end{aligned}
$$

分号分隔不同轮次，共 11 个坐标。累计核 $k^{(h)}_{\mathrm{WLsubtree}}$ 是向量内积，也就是把各轮贡献相加：

$$
k^{(h)}_{\mathrm{WLsubtree}}(G_A,G_C)=\sum_{i=0}^{h}K_i(A,C).
$$

我们再用余弦归一化得到 $S_h$：

$$
S_h(A,C)=
\frac{\sum_{i=0}^{h}K_i(A,C)}
{\sqrt{\left(\sum_{i=0}^{h}K_i(A,A)\right)
\left(\sum_{i=0}^{h}K_i(C,C)\right)}}.
$$

代入各轮结果：

$$
\begin{aligned}
S_0(A,C)&=\frac{10}{\sqrt{8\times13}}=0.9806,\\
S_1(A,C)&=\frac{10+6}{\sqrt{(8+8)(13+7)}}=0.8944,\\
S_2(A,C)&=\frac{10+6+2}{\sqrt{(8+8+8)(13+7+7)}}=\boxed{0.7071}.
\end{aligned}
$$

**不能只拿第 2 轮的 $2/\sqrt{8\times7}$ 作为 $S_2$，因为它遗漏了第 0、1 轮。** 分数越高，选定图的特征越接近；0.7071 不是约束重合百分比或求解准确率。

## 7. Bounds 与原论文的关系

如果把 C 新增的 $x_1\leq3$ 转入 Bounds，并排除 Bounds 节点，两图的连接和类型标签完全相同；每轮共同贡献与自身贡献都为 8，因此：

$$
S_2(A,C)=\frac{8+8+8}{\sqrt{(8+8+8)(8+8+8)}}=1.
$$

上界仍然限制可行解，只是不被这个图编码观察。因此，**排除 Bounds 不保证相似度降低；相似度为 1 也不证明数学模型等价。** 本报告的 0.7071 对应“共同非负界不入图、新增上界仍作显式行”的教学例子，不能当作“所有有限上下界均纳入约束”的统一计算结果。

标签更新遵循 [Shervashidze et al. (2011), *Weisfeiler–Lehman Graph Kernels*](https://jmlr.csail.mit.edu/papers/volume12/shervashidze11a/shervashidze11a.pdf) 的 Algorithm 1/2，累计计数内积对应 Definition 4。**LP 如何转成图及余弦归一化是本研究的应用选择。**

计算已与 [wl_bipartite.py](/Users/cora/Documents/GitHub/lean-llm-opt/wl_bipartite.py) 核对；数值记录见 [toy_verification.json](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_tutorial_20261006/toy_verification.json)，逐节点标签见 [node_labels_all_rounds.csv](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_tutorial_20261006/node_labels_all_rounds.csv)。
