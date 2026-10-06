## Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$, with business identifier "Product Name" from 41.csv.
- $x_i$ = production quantity of product $i$ (continuous, $x_i \geq 0$).

Parameters (from 41.csv, table_id: file_0_view_0):
- $l_i$ = "Labor per unit" required for product $i$.
- $m_i$ = "Material per unit" required for product $i$.
- $s_i$ = "Selling Price" per unit of product $i$.
- $v_i$ = "Variable Cost" per unit of product $i$.

Global resource capacities:
- Total available labor per week: $L = 1650$
- Total available material per week: $M = 1850$
- Fixed weekly operating cost: $F = 4500$

### Objective
Maximize weekly net profit:
$$
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
$$

### Constraints

1. Labor capacity:
$$
\sum_{i \in I} l_i x_i \leq L
$$

2. Material capacity:
$$
\sum_{i \in I} m_i x_i \leq M
$$

3. Nonnegativity:
$$
x_i \geq 0 \quad \forall i \in I
$$

### Data Mapping

- $I$ (products): "Product Name" (file_0_view_0)
- $l_i$: "Labor per unit" (file_0_view_0, column)
- $m_i$: "Material per unit" (file_0_view_0, column)
- $s_i$: "Selling Price" (file_0_view_0, column)
- $v_i$: "Variable Cost" (file_0_view_0, column)
- $L = 1650$, $M = 1850$, $F = 4500$ (from user description)

All 198 products and their coefficients are included as indexed by "Product Name" in 41.csv.