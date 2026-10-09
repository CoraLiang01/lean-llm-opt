Mathematical Model

Sets:
- $I$: set of products, indexed by $i$ (Product Name from file_0_view_0)

Parameters:
- $l_i$: labor required per unit of product $i$ (Labor per unit, file_0_view_0)
- $m_i$: material required per unit of product $i$ (Material per unit, file_0_view_0)
- $s_i$: selling price per unit of product $i$ (Selling Price, file_0_view_0)
- $v_i$: variable cost per unit of product $i$ (Variable Cost, file_0_view_0)
- $L$: total weekly labor capacity (given, $1650$)
- $M$: total weekly material capacity (given, $1850$)
- $F$: fixed weekly operating cost (given, $4500$)

Decision Variables:
- $x_i \geq 0$: continuous, production quantity of product $i$ (units)

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

- $I$: All rows of file_0_view_0, column Product Name
- $l_i$: file_0_view_0, column Labor per unit, keyed by Product Name
- $m_i$: file_0_view_0, column Material per unit, keyed by Product Name
- $s_i$: file_0_view_0, column Selling Price, keyed by Product Name
- $v_i$: file_0_view_0, column Variable Cost, keyed by Product Name
- $L = 1650$ (from user description)
- $M = 1850$ (from user description)
- $F = 4500$ (from user description)
- $x_i$: continuous, nonnegative, indexed by Product Name

All parameter values are to be taken directly from the specified columns and rows of file_0_view_0 (41.csv).