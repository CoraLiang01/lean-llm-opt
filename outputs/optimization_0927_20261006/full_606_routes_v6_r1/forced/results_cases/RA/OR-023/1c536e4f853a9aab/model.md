Let $x_i$ denote the number of units of product $i$ (with Product_Reference as below) to fulfill.

**Parameters:**

| Product_Reference         | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|--------------------------|-----------------|----------------|--------------------------|
| ELE-SMA-10000463         | 4.0             | 295            | 2000.0                   |
| ELE-SMA-10000487         | 14.0            | 1002           | 7000.0                   |
| ELE-SMA-10003333         | 14.0            | 958            | 7000.0                   |
| ELE-SMA-10009012         | 4.0             | 777            | 6000.0                   |
| ELE-SMA-10009999         | 4.0             | 271            | 2000.0                   |
| ELE-SMA-10011234         | 4.0             | 244            | 2000.0                   |
| ELE-SMA-10027456         | 14.0            | 990            | 7000.0                   |
| ELE-SMA-10028567         | 14.0            | 1000           | 7000.0                   |

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Objective:**
$$
\max \sum_{i} r_i x_i
$$

**Constraints:**
- Inventory and demand limits for each product:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i
$$

**Explicitly, for each product:**
\[
\begin{align*}
0 \leq x_{\text{ELE-SMA-10000463}} &\leq 295 \\
0 \leq x_{\text{ELE-SMA-10000487}} &\leq 1002 \\
0 \leq x_{\text{ELE-SMA-10003333}} &\leq 958 \\
0 \leq x_{\text{ELE-SMA-10009012}} &\leq 777 \\
0 \leq x_{\text{ELE-SMA-10009999}} &\leq 271 \\
0 \leq x_{\text{ELE-SMA-10011234}} &\leq 244 \\
0 \leq x_{\text{ELE-SMA-10027456}} &\leq 990 \\
0 \leq x_{\text{ELE-SMA-10028567}} &\leq 1000 \\
\end{align*}
\]

**Variable domains:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Where:**
- $x_i$ = units of product $i$ to fulfill
- $r_i$ = Revenue for product $i$ (see table)
- $d_i$ = Demand for product $i$ (see table)
- $s_i$ = Initial Inventory for product $i$ (see table)

**Maximize total revenue from fulfilling demand for all ‘ELE-S’ products, subject to inventory and demand limits.**