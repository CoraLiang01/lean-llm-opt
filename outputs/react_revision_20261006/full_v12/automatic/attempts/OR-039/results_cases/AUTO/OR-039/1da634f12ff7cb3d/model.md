### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), with identifiers ProductName from file_1_view_0.
- $J$ = set of warehouses (indexed by $j$), with identifiers Warehouse ID from file_0_view_0.

Parameters:
- $v_i$ = Value of vehicle type $i$ (file_1_view_0, column Value)
- $w_i$ = Weight (space requirement) of vehicle type $i$ (file_1_view_0, column Weight)
- $C_j$ = Capacity of warehouse $j$ (file_0_view_0, column Capacity)

Decision variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$ (integer, $x_{ij} \geq 0$)

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

- $I$: ProductName from file_1_view_0 (products.csv)
- $J$: Warehouse ID from file_0_view_0 (capacity.csv)
- $v_i$: Value from file_1_view_0, column Value
- $w_i$: Weight from file_1_view_0, column Weight
- $C_j$: Capacity from file_0_view_0, column Capacity
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)