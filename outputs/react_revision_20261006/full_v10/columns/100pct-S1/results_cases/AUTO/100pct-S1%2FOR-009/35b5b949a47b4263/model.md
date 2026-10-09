Mathematical Model

Index Sets:
- Let $I$ be the set of areas, with each $i \in I$ corresponding to a ProductName from file_1_view_0.

Parameters:
- $v_i$: Value (development benefit) of area $i$ (from file_1_view_0, column Value)
- $w_i$: Weight (resource usage per unit development in area $i$) (from file_1_view_0, column Weight)
- $C$: Overall development capacity (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$
$$
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
$$

Data Mapping

- $I$: All ProductName in file_1_view_0
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (area/ProductName)