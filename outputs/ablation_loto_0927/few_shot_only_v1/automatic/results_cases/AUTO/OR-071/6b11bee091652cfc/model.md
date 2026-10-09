Let $x_i$ denote the (nonnegative, continuous) number of units of product $i$ to produce per week, where $i$ indexes the 198 products in the order and with the names given in the table below.

**Parameters (from 41.csv):**

- $l_i$: Labor per unit for product $i$
- $m_i$: Material per unit for product $i$
- $s_i$: Selling price per unit for product $i$
- $v_i$: Variable cost per unit for product $i$

**Constants:**

- Total weekly labor available: $L = 1650$
- Total weekly material available: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

---

### Mathematical Model

**Decision variables:**
- $x_i \geq 0$ (continuous), for all products $i$ in the table below

**Objective:**
\[
\max \left( \sum_{i=1}^{198} (s_i - v_i) x_i - F \right)
\]

**Subject to:**
\[
\sum_{i=1}^{198} l_i x_i \leq L
\]
\[
\sum_{i=1}^{198} m_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i
\]

---

#### Product Index Table

| $i$ | Product Name                        | $l_i$ (Labor) | $m_i$ (Material) | $s_i$ (Selling Price) | $v_i$ (Variable Cost) |
|-----|-------------------------------------|---------------|------------------|-----------------------|-----------------------|
| 1   | Classic Oxford Shirt                | 3.1           | 4.2              | 125                   | 63                    |
| 2   | Cotton Crew Neck T-Shirt            | 2.1           | 3.1              | 83                    | 41                    |
| 3   | French Terry Hoodie                 | 6.5           | 6.8              | 195                   | 95                    |
| 4   | Denim Snap Shirt                    | 3.4           | 4.6              | 132                   | 68                    |
| 5   | Jersey V-Neck T-Shirt               | 2.5           | 3.4              | 90                    | 48                    |
| 6   | Unisex Jogger Pants                 | 5.9           | 6.1              | 185                   | 88                    |
| 7   | Flannel Plaid Shirt                 | 3.5           | 4                | 128                   | 65                    |
| 8   | Performance Polo Shirt              | 2.9           | 3.9              | 118                   | 57                    |
| 9   | Long-sleeve Henley                  | 2.8           | 3.7              | 95                    | 50                    |
| 10  | Heavyweight Sweatshirt              | 6.2           | 6.4              | 190                   | 92                    |
| ... | ...                                 | ...           | ...              | ...                   | ...                   |
| 198 | Heavyweight Denim Shirt             | 4             | 5.1              | 145                   | 77                    |

(For $i = 1, \ldots, 198$, the full list and coefficients are as in the table above, in the original CSV order.)

---

**Summary of the Model:**

\[
\begin{align*}
\max_{x_i \geq 0} \quad & \sum_{i=1}^{198} (s_i - v_i) x_i - 4500 \\
\text{s.t.} \quad & \sum_{i=1}^{198} l_i x_i \leq 1650 \\
                   & \sum_{i=1}^{198} m_i x_i \leq 1850 \\
                   & x_i \geq 0 \quad \forall i = 1, \ldots, 198
\end{align*}
\]

where all coefficients and product names are as given in the table above.