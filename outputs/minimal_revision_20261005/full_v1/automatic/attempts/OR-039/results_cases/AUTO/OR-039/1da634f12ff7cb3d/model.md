Abstract Mathematical Model

Sets:
- $I$: set of vehicle types, indexed by $i$ (from products.csv, column ProductName)
- $J$: set of warehouses, indexed by $j$ (from capacity.csv, column Warehouse ID)

Parameters:
- $v_i$: value (benefit coefficient) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: weight (space requirement) of vehicle type $i$ (from products.csv, column Weight)
- $C_j$: capacity of warehouse $j$ (from capacity.csv, column Capacity)

Decision Variables:
- $x_{ij}$: number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (vehicle types): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $J$ (warehouses): file_0_view_0, column Warehouse ID
- $C_j$: file_0_view_0, column Capacity, keyed by Warehouse ID

All parameters and sets are to be populated directly from the corresponding columns and rows in the returned CSVQA data, preserving original order and identifiers.