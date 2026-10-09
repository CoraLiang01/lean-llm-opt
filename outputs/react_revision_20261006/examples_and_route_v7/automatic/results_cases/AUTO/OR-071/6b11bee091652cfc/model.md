Mathematical Model

Sets:
- $I$: set of products (indexed by $i$), from 41.csv "Product Name"

Parameters:
- $a_i$: labor required per unit of product $i$ (41.csv, "Labor per unit")
- $b_i$: material required per unit of product $i$ (41.csv, "Material per unit")
- $s_i$: selling price per unit of product $i$ (41.csv, "Selling Price")
- $v_i$: variable cost per unit of product $i$ (41.csv, "Variable Cost")
- $L$: total weekly labor capacity $=1650$
- $M$: total weekly material capacity $=1850$
- $F$: fixed weekly operating cost $=4500$

Decision Variables:
- $x_i \geq 0$: continuous, production quantity of product $i$ (units)

Objective:
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

Subject to:
\[
\sum_{i \in I} a_i x_i \leq L
\]
\[
\sum_{i \in I} b_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

Data Mapping

- $I$: All rows in table_id file_0_view_0, column "Product Name"
- $a_i$: file_0_view_0, column "Labor per unit", keyed by "Product Name"
- $b_i$: file_0_view_0, column "Material per unit", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Selling Price", keyed by "Product Name"
- $v_i$: file_0_view_0, column "Variable Cost", keyed by "Product Name"
- $L$: 1650 (from user description)
- $M$: 1850 (from user description)
- $F$: 4500 (from user description)
- $x_i$: continuous, nonnegative, indexed by $i \in I$ ("Product Name")