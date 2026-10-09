##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from ProductName in products.csv
- $J$ = set of warehouses (indexed by $j$), from Warehouse ID in capacity.csv

Parameters:
- $v_i$ = value (benefit coefficient) of vehicle type $i$ (from Value in products.csv)
- $w_i$ = weight (space requirement) of vehicle type $i$ (from Weight in products.csv)
- $C_j$ = capacity of warehouse $j$ (from Capacity in capacity.csv)

Decision Variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j \quad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (vehicle types): ProductName in file_1_view_0 (products.csv)
- $v_i$: Value in file_1_view_0 (products.csv), mapped by ProductName
- $w_i$: Weight in file_1_view_0 (products.csv), mapped by ProductName
- $J$ (warehouses): Warehouse ID in file_0_view_0 (capacity.csv)
- $C_j$: Capacity in file_0_view_0 (capacity.csv), mapped by Warehouse ID
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)