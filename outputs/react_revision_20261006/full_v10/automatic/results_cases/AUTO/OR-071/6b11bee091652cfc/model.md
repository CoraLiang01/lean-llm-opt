Mathematical Model

Index Sets:
- Let $I$ be the set of all products, with each $i \in I$ corresponding to a unique "Product Name" from file_0_view_0.

Parameters:
- $l_i$: Labor per unit required for product $i$ (from "Labor per unit", file_0_view_0)
- $m_i$: Material per unit required for product $i$ (from "Material per unit", file_0_view_0)
- $s_i$: Selling price per unit of product $i$ (from "Selling Price", file_0_view_0)
- $v_i$: Variable cost per unit of product $i$ (from "Variable Cost", file_0_view_0)
- $L = 1650$: Total weekly labor capacity
- $M = 1850$: Total weekly material capacity
- $F = 4500$: Fixed weekly operating cost

Decision Variables:
- $x_i \geq 0$: Continuous production quantity of product $i$ to produce in the week

Objective:
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

Subject to:
\[
\sum_{i \in I} l_i x_i \leq L
\]
\[
\sum_{i \in I} m_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

Data Mapping

Index Sets:
- $I$: All "Product Name" in file_0_view_0

Parameters:
- $l_i$: "Labor per unit" in file_0_view_0 for product $i$
- $m_i$: "Material per unit" in file_0_view_0 for product $i$
- $s_i$: "Selling Price" in file_0_view_0 for product $i$
- $v_i$: "Variable Cost" in file_0_view_0 for product $i$
- $L = 1650$ (from user description)
- $M = 1850$ (from user description)
- $F = 4500$ (from user description)

Variables:
- $x_i$: Continuous, nonnegative, for each "Product Name" in file_0_view_0

Objective:
- Maximize total net profit: total sales revenue minus total variable cost minus fixed weekly operating cost

Constraints:
- Labor: $\sum_{i \in I} l_i x_i \leq L$
- Material: $\sum_{i \in I} m_i x_i \leq M$
- Nonnegativity: $x_i \geq 0$ for all $i \in I$