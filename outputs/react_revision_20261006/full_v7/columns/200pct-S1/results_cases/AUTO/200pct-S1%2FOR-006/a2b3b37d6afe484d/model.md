Mathematical Optimization Model

Index Sets:
- $I$: Set of vehicle types (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: Benefit coefficient of vehicle type $i$ (from file_1_view_0, column Value)
- $w_i$: Inventory weight (space requirement) of vehicle type $i$ (from file_1_view_0, column Weight)
- $C$: Total inventory capacity (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

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

Data Mapping:
- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$