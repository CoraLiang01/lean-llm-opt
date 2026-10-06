ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, indexed by $i$. (From products.csv, column ProductName)

Parameters:
- $v_i$: Profit from selling one unit of vehicle type $i$. (products.csv, column Value, key ProductName)
- $w_i$: Inventory space required by one unit of vehicle type $i$. (products.csv, column Weight, key ProductName)
- $C$: Total inventory capacity. (capacity.csv, column Capacity)

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

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

DATA MAPPING

- $I$: All records in products.csv, column ProductName, table_id: file_1_view_0
- $v_i$: products.csv, column Value, key ProductName, table_id: file_1_view_0
- $w_i$: products.csv, column Weight, key ProductName, table_id: file_1_view_0
- $C$: capacity.csv, column Capacity, table_id: file_0_view_0

All data is used as returned, preserving source order and identifiers.