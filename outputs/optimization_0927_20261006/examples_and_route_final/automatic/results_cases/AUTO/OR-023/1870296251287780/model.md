Let $x_i$ denote the number of units of product $i$ to fulfill, for each product $i$ classified as ‘ELE-S’ with Product_Reference as below.

**Parameters:**

- $r_i$: Revenue per unit of product $i$ (from ‘Revenue’ column)
- $d_i$: Demand for product $i$ (from ‘Demand’ column)
- $s_i$: Initial Inventory of product $i$ (from ‘Initial Inventory’ column)

**Product Indexing (in source order):**

| $i$ | Product_Reference         | $r_i$ | $d_i$ | $s_i$   |
|-----|--------------------------|-------|-------|---------|
| 1   | ELE-SMA-10000463         | 4.0   | 295   | 2000.0  |
| 2   | ELE-SMA-10000487         | 14.0  | 1002  | 7000.0  |
| 3   | ELE-SMA-10003333         | 14.0  | 958   | 7000.0  |
| 4   | ELE-SMA-10009012         | 4.0   | 777   | 6000.0  |
| 5   | ELE-SMA-10009999         | 4.0   | 271   | 2000.0  |
| 6   | ELE-SMA-10011234         | 4.0   | 244   | 2000.0  |
| 7   | ELE-SMA-10027456         | 14.0  | 990   | 7000.0  |
| 8   | ELE-SMA-10028567         | 14.0  | 1000  | 7000.0  |

**Decision Variables:**

$x_i \in \mathbb{Z}_{\geq 0}$ for all $i=1,\ldots,8$

---

**Mathematical Model:**

Objective:
\[
\max \sum_{i=1}^{8} r_i x_i
\]

Subject to, for all $i=1,\ldots,8$:
\[
x_i \leq d_i
\]
\[
x_i \leq s_i
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z}
\]

---

**Explicitly, with data:**

\[
\max \Big(
4.0\, x_1 + 14.0\, x_2 + 14.0\, x_3 + 4.0\, x_4 + 4.0\, x_5 + 4.0\, x_6 + 14.0\, x_7 + 14.0\, x_8
\Big)
\]

Subject to:
\[
\begin{align*}
x_1 &\leq 295 \\
x_1 &\leq 2000 \\
x_2 &\leq 1002 \\
x_2 &\leq 7000 \\
x_3 &\leq 958 \\
x_3 &\leq 7000 \\
x_4 &\leq 777 \\
x_4 &\leq 6000 \\
x_5 &\leq 271 \\
x_5 &\leq 2000 \\
x_6 &\leq 244 \\
x_6 &\leq 2000 \\
x_7 &\leq 990 \\
x_7 &\leq 7000 \\
x_8 &\leq 1000 \\
x_8 &\leq 7000 \\
x_i &\geq 0,\quad x_i \in \mathbb{Z},\quad \forall i=1,\ldots,8
\end{align*}
\]

Where $x_1$ corresponds to ELE-SMA-10000463, $x_2$ to ELE-SMA-10000487, ..., $x_8$ to ELE-SMA-10028567.