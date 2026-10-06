ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types (from products.csv, column ProductName)

Parameters:
- $v_i$: Profit per unit of vehicle $i$ (from products.csv, column Value, table_id: file_1_view_0)
- $w_i$: Inventory weight per unit of vehicle $i$ (from products.csv, column Weight, table_id: file_1_view_0)
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id: file_0_view_0)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day

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
- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C$: file_0_view_0 (capacity.csv), column Capacity

All data is used as returned, preserving original file and row order.