Mathematical Model

Index Sets:
- $I$: Set of vehicle types (from file_1_view_0, column ProductName)

Parameters:
- $p_i$: Profit per unit of vehicle $i$ (file_1_view_0, column Value, for each $i \in I$)
- $w_i$: Inventory weight per unit of vehicle $i$ (file_1_view_0, column Weight, for each $i \in I$)
- $C$: Total inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type from ProductName)