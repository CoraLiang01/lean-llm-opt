ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas available for development (from file_1_view_0, column ProductName).

Parameters:
- $v_i$: Development benefit per unit in area $i$ (from file_1_view_0, column Value, key ProductName).
- $w_i$: Resource requirement per unit in area $i$ (from file_1_view_0, column Weight, key ProductName).
- $C$: Overall development capacity (from file_0_view_0, column Capacity).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$.

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
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (area)