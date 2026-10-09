ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types (from file_1_view_0, column ProductName)

Parameters:
- $p_i$: Profit per unit of vehicle $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: Inventory weight per unit of vehicle $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: Total inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day

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

Data Mapping:
- $I$: All ProductName in file_1_view_0
- $p_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$