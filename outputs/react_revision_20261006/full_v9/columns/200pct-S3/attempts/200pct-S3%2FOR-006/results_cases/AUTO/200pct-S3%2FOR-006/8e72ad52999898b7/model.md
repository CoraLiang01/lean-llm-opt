ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: Value (benefit) of ordering one unit of vehicle type $i$ (from file_1_view_0, column Value)
- $w_i$: Weight (inventory space required) for one unit of vehicle type $i$ (from file_1_view_0, column Weight)
- $C$: Total inventory capacity (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ to order daily

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
- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type from file_1_view_0, column ProductName)