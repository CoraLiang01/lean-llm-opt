Mathematical Model

Index Sets:
- Let $I$ be the set of vehicle types, indexed by $i$ (from ProductName in file_1_view_0).
- Let $J$ be the set of warehouses, indexed by $j$ (from Warehouse ID in file_0_view_0).

Parameters:
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (Value from file_1_view_0, column Value, key ProductName).
- $w_i$: Space requirement (Weight) of vehicle type $i$ (Weight from file_1_view_0, column Weight, key ProductName).
- $C_j$: Capacity of warehouse $j$ (Capacity from file_0_view_0, column Capacity, key Warehouse ID).

Decision Variables:
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$. $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_i x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i x_{ij} \leq C_j \quad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $J$: All Warehouse ID in file_0_view_0 (capacity.csv)
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C_j$: file_0_view_0, column Capacity, key Warehouse ID
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)