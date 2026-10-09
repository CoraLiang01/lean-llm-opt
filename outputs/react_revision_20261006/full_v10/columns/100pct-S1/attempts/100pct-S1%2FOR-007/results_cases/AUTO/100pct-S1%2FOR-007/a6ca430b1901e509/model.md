ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $I$ be the set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
- $p_i$: Profit per unit of vehicle type $i$ (Value column, file_1_view_0).
- $w_i$: Inventory space required per unit of vehicle type $i$ (Weight column, file_1_view_0).
- $C$: Total inventory capacity (Capacity column, file_0_view_0).

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

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

DATA MAPPING

- $I$: All ProductName values from file_1_view_0.
- $p_i$: Value column from file_1_view_0, indexed by ProductName.
- $w_i$: Weight column from file_1_view_0, indexed by ProductName.
- $C$: Capacity column from file_0_view_0 (single value).
- $x_i$: Decision variable for each ProductName from file_1_view_0.