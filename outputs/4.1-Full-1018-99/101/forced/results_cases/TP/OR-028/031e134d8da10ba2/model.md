Let $I$ be the set of products:
$$
I = \{\text{sku\_I27},\ \text{sku\_I499},\ \text{sku\_I719},\ \text{sku\_T18},\ \text{sku\_T29},\ \text{sku\_T39},\ \text{sku\_T499},\ \text{sku\_T9},\ \text{sku\_3081},\ \text{sku\_339},\ \text{sku\_3799},\ \text{sku\_439},\ \text{sku\_539},\ \text{sku\_61399},\ \text{sku\_628},\ \text{sku\_708},\ \text{sku\_77},\ \text{sku\_79},\ \text{sku\_799},\ \text{sku\_8499},\ \text{sku\_89},\ \text{sku\_897},\ \text{sku\_9699},\ \text{sku\_bobo}\}
$$

Let $x_i$ be the number of units of product $i$ to fulfill, for each $i\in I$.

Parameters (for each $i\in I$):

- $r_i$: Revenue per unit
- $d_i$: Demand
- $s_i$: Initial Inventory

Data (in source order):

| Product Name | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|--------------|-----------------|---------------|---------------------------|
| sku_I27      | 238             | 6             | 30                        |
| sku_I499     | 287             | 4             | 20                        |
| sku_I719     | 268             | 16            | 80                        |
| sku_T18      | 318             | 14            | 70                        |
| sku_T29      | 207             | 4             | 20                        |
| sku_T39      | 258             | 32            | 160                       |
| sku_T499     | 249             | 8             | 40                        |
| sku_T9       | 227             | 2             | 10                        |
| sku_3081     | 198             | 10            | 50                        |
| sku_339      | 254             | 8             | 40                        |
| sku_3799     | 246             | 18            | 90                        |
| sku_439      | 258             | 2             | 10                        |
| sku_539      | 268             | 4             | 20                        |
| sku_61399    | 278             | 8             | 40                        |
| sku_628      | 268             | 2             | 10                        |
| sku_708      | 298             | 198           | 990                       |
| sku_77       | 258             | 32            | 160                       |
| sku_79       | 315             | 18            | 90                        |
| sku_799      | 264             | 570           | 2870                      |
| sku_8499     | 238             | 6             | 30                        |
| sku_89       | 258             | 26            | 130                       |
| sku_897      | 268             | 6             | 30                        |
| sku_9699     | 288             | 33            | 170                       |
| sku_bobo     | 228             | 33            | 170                       |

Mathematical Model:

Decision variables:
$$
x_i \geq 0,\quad \text{continuous, for all } i\in I
$$

Objective:
$$
\max \sum_{i\in I} r_i x_i
$$

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i\in I \\
& x_i \leq s_i, && \forall i\in I \\
& x_i \geq 0, && \forall i\in I
\end{align*}
\]

Where:

- $x_i$: number of units of product $i$ to fulfill (continuous, $0 \leq x_i \leq \min\{d_i, s_i\}$)
- $r_i$: revenue per unit of product $i$ (see table above)
- $d_i$: demand for product $i$ (see table above)
- $s_i$: initial inventory for product $i$ (see table above)

All data and identifiers are preserved in source order.