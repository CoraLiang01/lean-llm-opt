Let $x_i$ be the nonnegative continuous production quantity (units per week) of product $i$, where $i$ indexes the products in the order given below.

**Parameters (from 41.csv, in source order):**

| $i$ | Product Name                        | Labor per unit ($a_i$) | Material per unit ($b_i$) | Selling Price ($s_i$) | Variable Cost ($v_i$) |
|-----|-------------------------------------|------------------------|--------------------------|----------------------|----------------------|
| 1   | Classic Oxford Shirt                | 3.1                    | 4.2                      | 125                  | 63                   |
| 2   | Cotton Crew Neck T-Shirt            | 2.1                    | 3.1                      | 83                   | 41                   |
| 3   | French Terry Hoodie                 | 6.5                    | 6.8                      | 195                  | 95                   |
| 4   | Denim Snap Shirt                    | 3.4                    | 4.6                      | 132                  | 68                   |
| 5   | Jersey V-Neck T-Shirt               | 2.5                    | 3.4                      | 90                   | 48                   |
| 6   | Unisex Jogger Pants                 | 5.9                    | 6.1                      | 185                  | 88                   |
| 7   | Flannel Plaid Shirt                 | 3.5                    | 4                        | 128                  | 65                   |
| 8   | Performance Polo Shirt              | 2.9                    | 3.9                      | 118                  | 57                   |
| 9   | Long-sleeve Henley                  | 2.8                    | 3.7                      | 95                   | 50                   |
| 10  | Heavyweight Sweatshirt              | 6.2                    | 6.4                      | 190                  | 92                   |
| ... | ...                                 | ...                    | ...                      | ...                  | ...                  |
| 100 | Heavyweight Denim Shirt             | 4                      | 5.1                      | 145                  | 77                   |

(Continue for all 100 products in the order and with the coefficients as given in the data above.)

**Constants:**
- Weekly labor capacity: $L = 1650$
- Weekly material capacity: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

---

### Mathematical Model

**Decision variables:**
- $x_i \geq 0$ (continuous), for all products $i = 1, \ldots, 100$

**Objective:**
\[
\max \left( \sum_{i=1}^{100} (s_i - v_i) x_i - F \right)
\]
where $s_i$ is the selling price and $v_i$ is the variable cost per unit of product $i$.

**Constraints:**
\[
\sum_{i=1}^{100} a_i x_i \leq L
\]
\[
\sum_{i=1}^{100} b_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i = 1, \ldots, 100
\]

**Where:**
- $a_i$ = labor per unit for product $i$ (from "Labor per unit" column)
- $b_i$ = material per unit for product $i$ (from "Material per unit" column)
- $s_i$ = selling price per unit for product $i$ (from "Selling Price" column)
- $v_i$ = variable cost per unit for product $i$ (from "Variable Cost" column)
- $L = 1650$ (weekly labor capacity)
- $M = 1850$ (weekly material capacity)
- $F = 4500$ (fixed weekly operating cost)

**Explicitly:**

\[
\begin{align*}
\max_{x_i \geq 0} \quad & \sum_{i=1}^{100} (s_i - v_i) x_i - 4500 \\
\text{s.t.} \quad & \sum_{i=1}^{100} a_i x_i \leq 1650 \\
                  & \sum_{i=1}^{100} b_i x_i \leq 1850 \\
                  & x_i \geq 0 \quad \forall i = 1, \ldots, 100
\end{align*}
\]

Where all coefficients and product names are as given in the retrieved data, in the original order.