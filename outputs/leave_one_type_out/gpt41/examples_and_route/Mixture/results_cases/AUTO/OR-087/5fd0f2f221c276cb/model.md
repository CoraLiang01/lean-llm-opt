Let $x_i$ be the number of 100 kg units of product $i$ to produce in the month (integer, $x_i \geq 0$), and $y_i$ be a binary variable indicating whether product $i$'s production line is activated ($y_i \in \{0,1\}$).

Let $i$ index the products as given in the files (A1, A2, ..., A80).

Parameters (from the data):

- $d_i$: Maximum Demand (100 kg units) for product $i$
- $p_i$: Selling Price ($/100 kg$) for product $i$
- $c_i$: Production Cost ($/100 kg$) for product $i$
- $q_i$: Production Quota (max per day, 100 kg units) for product $i$
- $f_i$: Activation Cost ($) for product $i$
- $b_i$: Minimum Batch Size (100 kg units) for product $i$
- $T = 22$: Number of production days in the month

All parameters are as given in the retrieved data, with the following mapping for each product $i$:

| Product | $d_i$ | $p_i$ | $c_i$ | $q_i$ | $f_i$ | $b_i$ |
|---------|-------|-------|-------|-------|-------|-------|
| A1      | 5732  | 111   | 71.38 | 586   | 178539| 18    |
| A2      | 5607  | 81    | 45.02 | 329   | 157708| 25    |
| ...     | ...   | ...   | ...   | ...   | ...   | ...   |
| A80     | 4466  | 106   | 67.19 | 556   | 146842| 18    |

(Continue for all 80 products as in the data.)

---

Objective:
\[
\max \sum_{i=1}^{80} \left[ (p_i - c_i) x_i - f_i y_i \right]
\]

Subject to:

1. **Demand constraint** (cannot produce more than demand):
   \[
   x_i \leq d_i \qquad \forall i = 1,\ldots,80
   \]

2. **Production capacity constraint** (cannot produce more than total monthly capacity for each product):
   \[
   x_i \leq q_i \cdot T \qquad \forall i = 1,\ldots,80
   \]

3. **Minimum batch size and activation logic** (if any is produced, must meet minimum batch size and pay activation cost):
   \[
   x_i \geq b_i y_i \qquad \forall i = 1,\ldots,80
   \]
   \[
   x_i \leq (q_i \cdot T) y_i \qquad \forall i = 1,\ldots,80
   \]
   (This ensures $y_i = 1$ if $x_i > 0$, and $x_i = 0$ if $y_i = 0$.)

4. **Variable domains**:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,80
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i = 1,\ldots,80
   \]

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{80} \left[ (p_i - c_i) x_i - f_i y_i \right] \\
\text{s.t.} \quad
& x_i \leq d_i \qquad \forall i = 1,\ldots,80 \\
& x_i \leq q_i \cdot 22 \qquad \forall i = 1,\ldots,80 \\
& x_i \geq b_i y_i \qquad \forall i = 1,\ldots,80 \\
& x_i \leq (q_i \cdot 22) y_i \qquad \forall i = 1,\ldots,80 \\
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,80 \\
& y_i \in \{0,1\} \qquad \forall i = 1,\ldots,80 \\
\end{align*}
\]

Where all coefficients ($d_i$, $p_i$, $c_i$, $q_i$, $f_i$, $b_i$) are as given in the retrieved data for each product $i$ (A1 to A80), in the original file order.