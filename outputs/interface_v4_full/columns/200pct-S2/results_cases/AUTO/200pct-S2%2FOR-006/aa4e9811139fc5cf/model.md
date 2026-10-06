Abstract Mathematical Model

Sets:
- $I$: Set of vehicle types, indexed by $i$ (from products.csv, column ProductName).

Parameters:
- $v_i$: Benefit coefficient of vehicle type $i$ (from products.csv, column Value, table_id: file_1_view_0).
- $w_i$: Inventory weight (space requirement) of vehicle type $i$ (from products.csv, column Weight, table_id: file_1_view_0).
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id: file_0_view_0).

Decision Variables:
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$.

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

Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity, source_row 0.

All parameters and sets are to be used exactly as returned, preserving original file and row order.