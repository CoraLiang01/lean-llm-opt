#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), corresponding to all ProductName in file_1_view_0.
- $J$ = set of warehouses (indexed by $j$), corresponding to all Warehouse ID in file_0_view_0.

Parameters:
- $v_i$ = Value of vehicle type $i$ (from Value in file_1_view_0).
- $w_i$ = Weight (space requirement) of vehicle type $i$ (from Weight in file_1_view_0).
- $C_j$ = Capacity of warehouse $j$ (from Capacity in file_0_view_0).

Decision variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

---

#### Data Mapping

- $I$ (vehicle types): file_1_view_0, column ProductName
- $J$ (warehouses): file_0_view_0, column Warehouse ID
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C_j$: file_0_view_0, column Capacity, keyed by Warehouse ID
- $x_{ij}$: integer variable for each $(i, j)$ pair

All parameters and index sets are defined by the full set of records in the respective files as returned above.