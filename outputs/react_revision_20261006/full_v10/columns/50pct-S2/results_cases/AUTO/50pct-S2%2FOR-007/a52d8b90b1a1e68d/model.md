ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $I$ be the set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
- $p_i$: Profit per unit of vehicle type $i$ (Value column, file_1_view_0).
- $w_i$: Inventory space required per unit of vehicle type $i$ (Weight column, file_1_view_0).
- $C$: Total inventory capacity (Capacity column, file_0_view_0).

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

DATA MAPPING

- $I$: All ProductName values in file_1_view_0.
- $p_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are derived directly from the current CSVQA data.