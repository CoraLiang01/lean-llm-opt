Mathematical Model

Let $I$ be the set of all products, indexed by $i$ (from the "Product Name" column in 41.csv).

Parameters:
- $l_i$: labor required per unit of product $i$ ("Labor per unit", 41.csv)
- $m_i$: material required per unit of product $i$ ("Material per unit", 41.csv)
- $s_i$: selling price per unit of product $i$ ("Selling Price", 41.csv)
- $v_i$: variable cost per unit of product $i$ ("Variable Cost", 41.csv)
- $L = 1650$: total weekly labor capacity
- $M = 1850$: total weekly material capacity
- $F = 4500$: fixed weekly operating cost

Decision Variables:
- $x_i \geq 0$: continuous, quantity of product $i$ to produce (units)

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

- $I$: All records in 41.csv, column "Product Name", table_id file_0_view_0
- $l_i$: 41.csv, column "Labor per unit", table_id file_0_view_0, keyed by "Product Name"
- $m_i$: 41.csv, column "Material per unit", table_id file_0_view_0, keyed by "Product Name"
- $s_i$: 41.csv, column "Selling Price", table_id file_0_view_0, keyed by "Product Name"
- $v_i$: 41.csv, column "Variable Cost", table_id file_0_view_0, keyed by "Product Name"
- $L = 1650$: from user description
- $M = 1850$: from user description
- $F = 4500$: from user description
- $x_i$: decision variable for each $i \in I$ (continuous, nonnegative)