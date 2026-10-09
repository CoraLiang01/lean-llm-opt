ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas (from ProductName in file_1_view_0)

Parameters:
- $v_i$: Development benefit per unit in area $i$ (Value, file_1_view_0, ProductName $i$)
- $w_i$: Development resource required per unit in area $i$ (Weight, file_1_view_0, ProductName $i$)
- $C$: Overall development capacity (Capacity, file_0_view_0, single value)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\]

DATA MAPPING

- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, key ProductName $i$
- $w_i$: file_1_view_0, column Weight, key ProductName $i$
- $C$: file_0_view_0, column Capacity (single value)
- $x_i$: Decision variable for each $i \in I$ (area from ProductName)