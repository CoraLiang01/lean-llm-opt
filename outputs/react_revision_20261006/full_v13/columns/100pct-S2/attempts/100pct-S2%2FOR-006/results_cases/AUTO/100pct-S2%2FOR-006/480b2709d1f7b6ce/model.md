ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, indexed by $i$. (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: Benefit coefficient of vehicle type $i$. (from file_1_view_0, column Value)
- $w_i$: Inventory weight (resource consumption) per unit of vehicle type $i$. (from file_1_view_0, column Weight)
- $C$: Total inventory capacity. (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of units of vehicle type $i$ to order daily. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

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
- $I$ (vehicle types): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

All parameters and index sets are defined directly from the current CSV data. The model maximizes total benefit from daily vehicle orders, subject to the overall inventory capacity, with integer, nonnegative order quantities for each vehicle type.