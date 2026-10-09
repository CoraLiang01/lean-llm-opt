Let $x_i$ be the nonnegative continuous production quantity (units) of product $i$ (where $i$ indexes the products in the order and with the names as given in the table below).

**Parameters (from 41.csv, source order):**

| $i$ | Product Name                      | Labor per unit ($a_i$) | Material per unit ($b_i$) | Selling Price ($p_i$) | Variable Cost ($v_i$) |
|-----|-----------------------------------|------------------------|--------------------------|----------------------|----------------------|
| 1   | Classic Oxford Shirt              | 3.1                    | 4.2                      | 125                  | 63                   |
| 2   | Cotton Crew Neck T-Shirt          | 2.1                    | 3.1                      | 83                   | 41                   |
| 3   | French Terry Hoodie               | 6.5                    | 6.8                      | 195                  | 95                   |
| 4   | Denim Snap Shirt                  | 3.4                    | 4.6                      | 132                  | 68                   |
| 5   | Jersey V-Neck T-Shirt             | 2.5                    | 3.4                      | 90                   | 48                   |
| 6   | Unisex Jogger Pants               | 5.9                    | 6.1                      | 185                  | 88                   |
| 7   | Flannel Plaid Shirt               | 3.5                    | 4                        | 128                  | 65                   |
| 8   | Performance Polo Shirt            | 2.9                    | 3.9                      | 118                  | 57                   |
| 9   | Long-sleeve Henley                | 2.8                    | 3.7                      | 95                   | 50                   |
| 10  | Heavyweight Sweatshirt            | 6.2                    | 6.4                      | 190                  | 92                   |
| ... | ...                               | ...                    | ...                      | ...                  | ...                  |
| 100 | Heavyweight Denim Shirt           | 4                      | 5.1                      | 145                  | 77                   |

(The full list of 100 products is as given in the retrieved data, in source order.)

**Constants:**
- Weekly labor capacity: $L = 1650$
- Weekly material capacity: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

---

### Mathematical Model

**Decision Variables:**
- $x_i \geq 0$ (continuous), for each product $i$ (as listed above).

**Objective:**
\[
\max \left( \sum_{i=1}^{100} (p_i - v_i) x_i - F \right)
\]
where $p_i$ is the selling price and $v_i$ is the variable cost per unit of product $i$.

**Constraints:**

1. **Labor Capacity:**
   \[
   \sum_{i=1}^{100} a_i x_i \leq L
   \]
   where $a_i$ is the labor per unit for product $i$.

2. **Material Capacity:**
   \[
   \sum_{i=1}^{100} b_i x_i \leq M
   \]
   where $b_i$ is the material per unit for product $i$.

3. **Nonnegativity:**
   \[
   x_i \geq 0 \qquad \forall i = 1, \ldots, 100
   \]

---

**Explicitly, with all coefficients (for the first few products as an example):**

\[
\max \left(
\begin{aligned}
& (125-63)x_1 + (83-41)x_2 + (195-95)x_3 + (132-68)x_4 + (90-48)x_5 \\
& + (185-88)x_6 + (128-65)x_7 + (118-57)x_8 + (95-50)x_9 + (190-92)x_{10} + \cdots \\
& - 4500
\end{aligned}
\right)
\]

Subject to:
\[
\begin{aligned}
& 3.1x_1 + 2.1x_2 + 6.5x_3 + 3.4x_4 + 2.5x_5 + 5.9x_6 + 3.5x_7 + 2.9x_8 + 2.8x_9 + 6.2x_{10} + \cdots \leq 1650 \\
& 4.2x_1 + 3.1x_2 + 6.8x_3 + 4.6x_4 + 3.4x_5 + 6.1x_6 + 4x_7 + 3.9x_8 + 3.7x_9 + 6.4x_{10} + \cdots \leq 1850 \\
& x_i \geq 0 \qquad \forall i = 1, \ldots, 100
\end{aligned}
\]

Where all coefficients and product names are as given in the retrieved data, in source order.