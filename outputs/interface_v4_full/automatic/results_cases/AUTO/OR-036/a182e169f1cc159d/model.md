ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, with business identifier ProductName from products.csv.

Parameters:
- $v_i$: Value (benefit coefficient) of vehicle type $i \in I$ (from products.csv, column Value, table_id: file_1_view_0).
- $w_i$: Weight (inventory space required per unit) of vehicle type $i \in I$ (from products.csv, column Weight, table_id: file_1_view_0).
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id: file_0_view_0).

Decision Variables:
- $x_i$: Number of units of vehicle type $i \in I$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$.

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

Data Mapping:

- $I$ (vehicle types): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

All parameters and identifiers are used exactly as returned by CSVQA.