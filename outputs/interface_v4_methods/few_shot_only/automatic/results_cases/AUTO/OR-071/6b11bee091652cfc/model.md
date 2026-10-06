## Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$, with business identifier "Product Name" from 41.csv.
- $x_i$ = production quantity of product $i$ (continuous, $x_i \geq 0$).

Parameters (from 41.csv, see Data Mapping below):
- $l_i$ = "Labor per unit" required for product $i$
- $m_i$ = "Material per unit" required for product $i$
- $s_i$ = "Selling Price" per unit of product $i$
- $v_i$ = "Variable Cost" per unit of product $i$
- $L = 1650$ (weekly labor capacity)
- $M = 1850$ (weekly material capacity)
- $F = 4500$ (fixed weekly operating cost)

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

---

## Data Mapping

- $I$: All records in 41.csv, column "Product Name", table_id: file_0_view_0
- $l_i$: 41.csv, column "Labor per unit", table_id: file_0_view_0, keyed by "Product Name"
- $m_i$: 41.csv, column "Material per unit", table_id: file_0_view_0, keyed by "Product Name"
- $s_i$: 41.csv, column "Selling Price", table_id: file_0_view_0, keyed by "Product Name"
- $v_i$: 41.csv, column "Variable Cost", table_id: file_0_view_0, keyed by "Product Name"
- $L$: 1650 (from user description)
- $M$: 1850 (from user description)
- $F$: 4500 (from user description)

All 198 products from 41.csv are included, with their exact coefficients and identifiers as provided.