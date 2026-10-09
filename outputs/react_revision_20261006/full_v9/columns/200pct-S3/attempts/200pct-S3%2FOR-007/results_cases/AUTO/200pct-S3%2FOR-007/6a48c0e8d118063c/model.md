ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: Profit per unit of vehicle $i$ (file_1_view_0, column Value, for $i \in I$)
- $w_i$: Inventory weight per unit of vehicle $i$ (file_1_view_0, column Weight, for $i \in I$)
- $C$: Total inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$, for $i \in I$)

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

DATA MAPPING

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)