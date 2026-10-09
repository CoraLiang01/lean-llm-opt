ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of vehicle types (from file_1_view_0, column ProductName)
- $J$: set of warehouses (from file_0_view_0, column Warehouse ID)

Parameters:
- $v_i$: value (benefit coefficient) of vehicle type $i$ (from file_1_view_0, column Value)
- $w_i$: weight (space requirement) of vehicle type $i$ (from file_1_view_0, column Weight)
- $C_j$: capacity of warehouse $j$ (from file_0_view_0, column Capacity)

Decision Variables:
- $x_{ij}$: number of units of vehicle type $i$ to store in warehouse $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping:
- $I$ (vehicle types): file_1_view_0, column ProductName
- $J$ (warehouses): file_0_view_0, column Warehouse ID
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C_j$: file_0_view_0, column Capacity, keyed by Warehouse ID
- $x_{ij}$: number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, indexed by $I$ and $J$)