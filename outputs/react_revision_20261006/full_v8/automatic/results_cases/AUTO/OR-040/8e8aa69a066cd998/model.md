Mathematical Model

Index Sets:
- $I$: set of areas (from file_1_view_0, column ProductName)

Parameters:
- $b_i$: benefit coefficient for area $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: development unit weight for area $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: overall development capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: integer, daily scale of development in area $i$; $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: file_1_view_0, column ProductName
- $b_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$