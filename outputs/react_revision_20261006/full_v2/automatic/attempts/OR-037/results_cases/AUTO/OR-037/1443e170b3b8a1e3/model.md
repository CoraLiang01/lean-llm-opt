Mathematical Model

Sets:
- $I$: Set of vehicle types (indexed by $i$), from all ProductName in file_1_view_0.

Parameters:
- $p_i$: Profit per unit of vehicle $i$ (Value, from file_1_view_0, column Value, key ProductName).
- $w_i$: Inventory space required per unit of vehicle $i$ (Weight, from file_1_view_0, column Weight, key ProductName).
- $C$: Total inventory capacity (Capacity, from file_0_view_0, single value).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

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

- $I$: All ProductName in file_1_view_0 (products.csv), preserve source order.
- $p_i$: file_1_view_0, column Value, key ProductName.
- $w_i$: file_1_view_0, column Weight, key ProductName.
- $C$: file_0_view_0, column Capacity (capacity.csv), single value.
- $x_i$: Decision variable for each $i \in I$.