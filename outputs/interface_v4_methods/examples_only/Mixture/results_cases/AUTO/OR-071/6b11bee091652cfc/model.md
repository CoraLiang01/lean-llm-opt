## Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business identifier "Product Name" from 41.csv)
- $x_i$ = production quantity of product $i$ (continuous, $x_i \geq 0$)

Parameters (from 41.csv, for each $i \in I$):
- $l_i$ = "Labor per unit" of product $i$
- $m_i$ = "Material per unit" of product $i$
- $s_i$ = "Selling Price" of product $i$
- $v_i$ = "Variable Cost" of product $i$

Global parameters:
- $L = 1650$ (weekly labor capacity)
- $M = 1850$ (weekly material capacity)
- $F = 4500$ (fixed weekly operating cost)

### Objective:
$$
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
$$

### Constraints:
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

- $I$: All records in 41.csv, column "Product Name"
- $l_i$: 41.csv, column "Labor per unit", key "Product Name"
- $m_i$: 41.csv, column "Material per unit", key "Product Name"
- $s_i$: 41.csv, column "Selling Price", key "Product Name"
- $v_i$: 41.csv, column "Variable Cost", key "Product Name"
- $L$: 1650 (from user description)
- $M$: 1850 (from user description)
- $F$: 4500 (from user description)

Variables:
- $x_i$: continuous, $\geq 0$, for each $i \in I$ ("Product Name" in 41.csv)