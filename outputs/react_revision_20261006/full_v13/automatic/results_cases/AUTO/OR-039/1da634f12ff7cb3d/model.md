## Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), corresponding to ProductName in file_1_view_0.
- $J$ = set of warehouses (indexed by $j$), corresponding to Warehouse ID in file_0_view_0.

Parameters:
- $v_i$ = Value of vehicle type $i$ (from Value in file_1_view_0).
- $w_i$ = Weight (space requirement) of vehicle type $i$ (from Weight in file_1_view_0).
- $C_j$ = Capacity of warehouse $j$ (from Capacity in file_0_view_0).

Decision variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$ (integer, $\geq 0$).

### Objective
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

### Constraints

1. **Warehouse Capacity Constraints** (for each warehouse $j$):
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j \quad \forall j \in J
   \]

2. **Non-negativity and Integrality**:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $J$: All Warehouse ID in file_0_view_0 (capacity.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C_j$: Capacity column in file_0_view_0, keyed by Warehouse ID
- $x_{ij}$: Decision variable for number of units of vehicle type $i$ in warehouse $j$ (integer, $\geq 0$)