Mathematical Model

Index Sets:
- $I$: Set of vehicle types (from file_1_view_0.ProductName)

Parameters:
- $p_i$: Profit per unit of vehicle $i$ (from file_1_view_0.Value)
- $w_i$: Weight (inventory space required) per unit of vehicle $i$ (from file_1_view_0.Weight)
- $C$: Total inventory capacity (from file_0_view_0.Capacity)

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

Data Mapping

- $I$: All records in file_1_view_0.ProductName
- $p_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C$: file_0_view_0.Capacity (single value)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)