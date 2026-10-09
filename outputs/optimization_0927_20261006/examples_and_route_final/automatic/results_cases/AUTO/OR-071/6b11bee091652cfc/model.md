Let $x_i$ be the nonnegative continuous production quantity (units) of product $i$ (where $i$ indexes the products in the order given below).

Parameters (from 41.csv, in source order):

| $i$ | Product Name                      | Labor per unit ($a_i$) | Material per unit ($b_i$) | Selling Price ($s_i$) | Variable Cost ($v_i$) |
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

(Continue for all 100 products in the order above; all coefficients are as in the retrieved data.)

Fixed weekly operating cost: $F = 4,\!500$

Labor capacity: $L = 1,\!650$

Material capacity: $M = 1,\!850$

---

**Mathematical Model**

**Decision variables:**
$$
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i = 1, \ldots, 100
$$

**Objective:**
$$
\max \left\{ \sum_{i=1}^{100} (s_i - v_i) x_i - F \right\}
$$

**Subject to:**
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

- $x_i$ = production quantity of product $i$ (continuous, $\geq 0$)
- $a_i$ = labor per unit for product $i$ (from "Labor per unit" column)
- $b_i$ = material per unit for product $i$ (from "Material per unit" column)
- $s_i$ = selling price per unit for product $i$ (from "Selling Price" column)
- $v_i$ = variable cost per unit for product $i$ (from "Variable Cost" column)
- $F = 4,\!500$ (fixed weekly operating cost)
- $L = 1,\!650$ (weekly labor capacity)
- $M = 1,\!850$ (weekly material capacity)

**All coefficients and product names are as in the retrieved data, in the original file order.**