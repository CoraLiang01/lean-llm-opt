ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of areas available for development (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: development benefit per unit in area $i$ (from file_1_view_0, column Value)
- $w_i$: development capacity required per unit in area $i$ (from file_1_view_0, column Weight)
- $C$: overall development capacity (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: scale of development per day in area $i$

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

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, matched by ProductName
- $w_i$: Weight column in file_1_view_0, matched by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (area/ProductName)