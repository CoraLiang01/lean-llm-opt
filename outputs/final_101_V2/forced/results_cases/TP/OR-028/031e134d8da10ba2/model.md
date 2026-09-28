Let $I$ be the set of products:
$$
I = \{\text{sku\_I27},\ \text{sku\_I499},\ \text{sku\_I719},\ \text{sku\_T18},\ \text{sku\_T29},\ \text{sku\_T39},\ \text{sku\_T499},\ \text{sku\_T9},\ \text{sku\_3081},\ \text{sku\_339},\ \text{sku\_3799},\ \text{sku\_439},\ \text{sku\_539},\ \text{sku\_61399},\ \text{sku\_628},\ \text{sku\_708},\ \text{sku\_77},\ \text{sku\_79},\ \text{sku\_799},\ \text{sku\_8499},\ \text{sku\_89},\ \text{sku\_897},\ \text{sku\_9699},\ \text{sku\_bobo}\}
$$

Decision variables:
$$
x_i \geq 0 \quad \text{(integer or continuous, as appropriate)} \\
\text{Number of units of product } i \text{ to fulfill, for each } i \in I
$$

Parameters (for each $i \in I$):

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|---------------|-----------------|---------------|---------------------------|
| sku_I27       | 238             | 6             | 30                        |
| sku_I499      | 287             | 4             | 20                        |
| sku_I719      | 268             | 16            | 80                        |
| sku_T18       | 318             | 14            | 70                        |
| sku_T29       | 207             | 4             | 20                        |
| sku_T39       | 258             | 32            | 160                       |
| sku_T499      | 249             | 8             | 40                        |
| sku_T9        | 227             | 2             | 10                        |
| sku_3081      | 198             | 10            | 50                        |
| sku_339       | 254             | 8             | 40                        |
| sku_3799      | 246             | 18            | 90                        |
| sku_439       | 258             | 2             | 10                        |
| sku_539       | 268             | 4             | 20                        |
| sku_61399     | 278             | 8             | 40                        |
| sku_628       | 268             | 2             | 10                        |
| sku_708       | 298             | 198           | 990                       |
| sku_77        | 258             | 32            | 160                       |
| sku_79        | 315             | 18            | 90                        |
| sku_799       | 264             | 570           | 2870                      |
| sku_8499      | 238             | 6             | 30                        |
| sku_89        | 258             | 26            | 130                       |
| sku_897       | 268             | 6             | 30                        |
| sku_9699      | 288             | 33            | 170                       |
| sku_bobo      | 228             | 33            | 170                       |

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
\[
\begin{align*}
& 0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I \\
\end{align*}
\]

Explicitly, for each product $i$:

- $x_{\text{sku\_I27}} \leq 6$, $x_{\text{sku\_I27}} \leq 30$
- $x_{\text{sku\_I499}} \leq 4$, $x_{\text{sku\_I499}} \leq 20$
- $x_{\text{sku\_I719}} \leq 16$, $x_{\text{sku\_I719}} \leq 80$
- $x_{\text{sku\_T18}} \leq 14$, $x_{\text{sku\_T18}} \leq 70$
- $x_{\text{sku\_T29}} \leq 4$, $x_{\text{sku\_T29}} \leq 20$
- $x_{\text{sku\_T39}} \leq 32$, $x_{\text{sku\_T39}} \leq 160$
- $x_{\text{sku\_T499}} \leq 8$, $x_{\text{sku\_T499}} \leq 40$
- $x_{\text{sku\_T9}} \leq 2$, $x_{\text{sku\_T9}} \leq 10$
- $x_{\text{sku\_3081}} \leq 10$, $x_{\text{sku\_3081}} \leq 50$
- $x_{\text{sku\_339}} \leq 8$, $x_{\text{sku\_339}} \leq 40$
- $x_{\text{sku\_3799}} \leq 18$, $x_{\text{sku\_3799}} \leq 90$
- $x_{\text{sku\_439}} \leq 2$, $x_{\text{sku\_439}} \leq 10$
- $x_{\text{sku\_539}} \leq 4$, $x_{\text{sku\_539}} \leq 20$
- $x_{\text{sku\_61399}} \leq 8$, $x_{\text{sku\_61399}} \leq 40$
- $x_{\text{sku\_628}} \leq 2$, $x_{\text{sku\_628}} \leq 10$
- $x_{\text{sku\_708}} \leq 198$, $x_{\text{sku\_708}} \leq 990$
- $x_{\text{sku\_77}} \leq 32$, $x_{\text{sku\_77}} \leq 160$
- $x_{\text{sku\_79}} \leq 18$, $x_{\text{sku\_79}} \leq 90$
- $x_{\text{sku\_799}} \leq 570$, $x_{\text{sku\_799}} \leq 2870$
- $x_{\text{sku\_8499}} \leq 6$, $x_{\text{sku\_8499}} \leq 30$
- $x_{\text{sku\_89}} \leq 26$, $x_{\text{sku\_89}} \leq 130$
- $x_{\text{sku\_897}} \leq 6$, $x_{\text{sku\_897}} \leq 30$
- $x_{\text{sku\_9699}} \leq 33$, $x_{\text{sku\_9699}} \leq 170$
- $x_{\text{sku\_bobo}} \leq 33$, $x_{\text{sku\_bobo}} \leq 170$

Or, equivalently:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

All parameters and identifiers are as retrieved and preserved in source order.