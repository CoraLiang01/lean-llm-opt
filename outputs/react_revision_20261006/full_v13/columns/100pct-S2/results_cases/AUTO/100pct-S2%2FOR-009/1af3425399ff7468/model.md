ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas (from file_1_view_0.ProductName)

Parameters:
- $v_i$: Development benefit per unit in area $i$ (from file_1_view_0.Value)
- $w_i$: Resource requirement per unit in area $i$ (from file_1_view_0.Weight)
- $C$: Total development capacity (from file_0_view_0.Capacity)

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

Data Mapping:
- $I$: All file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C$: file_0_view_0.Capacity
- $x_i$: Decision variable for each $i \in I$