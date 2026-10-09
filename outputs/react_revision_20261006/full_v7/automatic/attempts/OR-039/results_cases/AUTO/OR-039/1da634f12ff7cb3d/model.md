#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from ProductName in file_1_view_0
- $J$ = set of warehouses (indexed by $j$), from Warehouse ID in file_0_view_0

Parameters:
- $v_i$ = value (benefit coefficient) of vehicle type $i$ (Value, file_1_view_0)
- $w_i$ = weight (space requirement) of vehicle type $i$ (Weight, file_1_view_0)
- $C_j$ = capacity of warehouse $j$ (Capacity, file_0_view_0)

Decision variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i x_{ij} \leq C_j \quad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $J$: All Warehouse ID in file_0_view_0 (capacity.csv)
- $v_i$: Value column, file_1_view_0, keyed by ProductName
- $w_i$: Weight column, file_1_view_0, keyed by ProductName
- $C_j$: Capacity column, file_0_view_0, keyed by Warehouse ID
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)