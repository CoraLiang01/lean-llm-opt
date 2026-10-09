Mathematical Model

Index Sets:
- $I$: set of vehicle types, with elements $i$ corresponding to ProductName in file_1_view_0.

Parameters:
- $v_i$: Value of vehicle type $i$ (file_1_view_0, column Value, key ProductName).
- $w_i$: Weight (inventory space required) of vehicle type $i$ (file_1_view_0, column Weight, key ProductName).
- $C$: Total inventory capacity (file_0_view_0, column Capacity).

Decision Variables:
- $x_i$: number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv)
- $x_i$: integer, for each $i \in I$