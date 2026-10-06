Abstract Mathematical Model

Sets:
- $I$: Set of vehicle types, indexed by $i$ (from ProductName in products.csv).

Parameters:
- $v_i$: Benefit coefficient of vehicle type $i$ (Value column in products.csv, table_id: file_1_view_0).
- $w_i$: Inventory weight (space requirement) of vehicle type $i$ (Weight column in products.csv, table_id: file_1_view_0).
- $C$: Total inventory capacity (Capacity column in capacity.csv, table_id: file_0_view_0).

Decision Variables:
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$ (integer, nonnegative).

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

- $I$: All rows in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity (single value, row 0).

All parameters and sets are to be populated directly from the specified columns and rows in the returned CSVQA data.