ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of vehicle types (indexed by $i$), from file_1_view_0["ProductName"]
- $J$: set of warehouses (indexed by $j$), from file_0_view_0["Warehouse ID"]

Parameters:
- $v_i$: value per unit of vehicle type $i$, from file_1_view_0["Value"]
- $w_i$: weight (space requirement) per unit of vehicle type $i$, from file_1_view_0["Weight"]
- $C_j$: capacity of warehouse $j$, from file_0_view_0["Capacity"]

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of vehicle type $i$ to store in warehouse $j$

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

DATA MAPPING

- $I$ (vehicle types): file_1_view_0["ProductName"]
- $J$ (warehouses): file_0_view_0["Warehouse ID"]
- $v_i$: file_1_view_0["Value"], keyed by file_1_view_0["ProductName"]
- $w_i$: file_1_view_0["Weight"], keyed by file_1_view_0["ProductName"]
- $C_j$: file_0_view_0["Capacity"], keyed by file_0_view_0["Warehouse ID"]
- $x_{ij}$: integer, nonnegative, for all $i \in I$, $j \in J$