ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of products (Product Name from file_0_view_0)

Parameters:
- $l_i$: labor required per unit of product $i$ (Labor per unit, file_0_view_0)
- $m_i$: material required per unit of product $i$ (Material per unit, file_0_view_0)
- $s_i$: selling price per unit of product $i$ (Selling Price, file_0_view_0)
- $v_i$: variable cost per unit of product $i$ (Variable Cost, file_0_view_0)
- $L$: total weekly labor capacity $=1650$
- $M$: total weekly material capacity $=1850$
- $F$: fixed weekly operating cost $=4500$

Decision Variables:
- $x_i \geq 0$: continuous production quantity of product $i$ (for all $i \in I$)

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

DATA MAPPING

- $I$: Product Name (file_0_view_0)
- $l_i$: Labor per unit (file_0_view_0, column "Labor per unit")
- $m_i$: Material per unit (file_0_view_0, column "Material per unit")
- $s_i$: Selling Price (file_0_view_0, column "Selling Price")
- $v_i$: Variable Cost (file_0_view_0, column "Variable Cost")
- $L$: 1650 (from user description)
- $M$: 1850 (from user description)
- $F$: 4500 (from user description)
- $x_i$: continuous, nonnegative, for each Product Name in file_0_view_0

All parameters are mapped directly from the columns of file_0_view_0 as specified.