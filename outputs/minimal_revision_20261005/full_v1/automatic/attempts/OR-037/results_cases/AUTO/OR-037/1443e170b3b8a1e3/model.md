Abstract Mathematical Model

Sets:
- $I$: Set of vehicle types (indexed by $i$), corresponding to each row in products.csv.

Parameters:
- $p_i$: Profit per unit of vehicle type $i$ (from products.csv, column Value, table_id file_1_view_0).
- $w_i$: Inventory space required per unit of vehicle type $i$ (from products.csv, column Weight, table_id file_1_view_0).
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id file_0_view_0).

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

Data Mapping

- $I$: Each record in file_1_view_0 (products.csv), column ProductName.
- $p_i$: file_1_view_0 (products.csv), column Value, for each $i$.
- $w_i$: file_1_view_0 (products.csv), column Weight, for each $i$.
- $C$: file_0_view_0 (capacity.csv), column Capacity.
- $x_i$: Decision variable for each $i$ in $I$.

All parameters and sets are mapped directly from the returned CSVQA data, preserving original row and column identifiers.