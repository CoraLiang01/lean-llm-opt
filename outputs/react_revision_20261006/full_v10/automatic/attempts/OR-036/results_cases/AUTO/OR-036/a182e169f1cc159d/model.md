Mathematical Model

Index Sets:
- Let $I$ be the set of vehicle types, with each $i \in I$ corresponding to a ProductName from file_1_view_0.

Parameters:
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from file_1_view_0, column Value).
- $w_i$: Weight (inventory space required per unit) of vehicle type $i$ (from file_1_view_0, column Weight).
- $C$: Total inventory capacity (from file_0_view_0, column Capacity).

Decision Variables:
- $x_i$: Number of units of vehicle type $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$.

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

- $I$: All ProductName in file_1_view_0 (products.csv, column ProductName)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)