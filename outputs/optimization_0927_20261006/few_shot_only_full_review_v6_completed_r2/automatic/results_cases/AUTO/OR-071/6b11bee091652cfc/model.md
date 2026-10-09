Let $x_i \geq 0$ be the production quantity (in units) of product $i$, for each product listed in 41.csv.

**Parameters (from 41.csv, all 198 products, in file order):**

For each product $i$:
- $l_i$ = Labor per unit (from "Labor per unit")
- $m_i$ = Material per unit (from "Material per unit")
- $s_i$ = Selling Price (from "Selling Price")
- $v_i$ = Variable Cost (from "Variable Cost")

**Constants:**
- Total weekly labor capacity: $L = 1650$
- Total weekly material capacity: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

**Model:**

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
x_i \geq 0 \quad \forall i \in \{1,2,\ldots,198\}
\]

**Where:**

For each $i$ (using the exact file order and identifiers):

| Product Name                      | $l_i$ (Labor per unit) | $m_i$ (Material per unit) | $s_i$ (Selling Price) | $v_i$ (Variable Cost) |
|-----------------------------------|------------------------|---------------------------|-----------------------|----------------------|
| Classic Oxford Shirt              | 3.1                    | 4.2                       | 125                   | 63                   |
| Cotton Crew Neck T-Shirt          | 2.1                    | 3.1                       | 83                    | 41                   |
| French Terry Hoodie               | 6.5                    | 6.8                       | 195                   | 95                   |
| Denim Snap Shirt                  | 3.4                    | 4.6                       | 132                   | 68                   |
| Jersey V-Neck T-Shirt             | 2.5                    | 3.4                       | 90                    | 48                   |
| ...                               | ...                    | ...                       | ...                   | ...                  |
| Heavyweight Denim Shirt           | 4                      | 5.1                       | 145                   | 77                   |

(Continue for all 198 products in the order and with the values as in 41.csv.)

**Decision variables:**
\[
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i
\]

**Summary of constraints:**
- Total labor used by all products cannot exceed 1,650 units.
- Total material used by all products cannot exceed 1,850 units.
- Production quantities are nonnegative and continuous.

**Net profit is total sales revenue minus total variable cost minus the fixed weekly operating cost.**