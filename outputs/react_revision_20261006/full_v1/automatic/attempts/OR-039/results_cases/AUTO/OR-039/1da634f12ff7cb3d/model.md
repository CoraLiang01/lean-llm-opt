### Symbolic Mathematical Model

**Sets:**
- $I$: set of vehicle types (indexed by $i$), from all ProductName in products.csv
- $J$: set of warehouses (indexed by $j$), from all Warehouse ID in capacity.csv

**Parameters:**
- $v_i$: value (benefit coefficient) of vehicle type $i$
- $w_i$: weight (space requirement) of vehicle type $i$
- $C_j$: capacity of warehouse $j$

**Decision Variables:**
- $x_{ij}$: number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \, x_{ij}
\]

**Constraints:**
1. **Warehouse Capacity Constraints:**
   \[
   \sum_{i \in I} w_i \, x_{ij} \leq C_j \qquad \forall j \in J
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$ (vehicle types): all ProductName in `file_1_view_0` (products.csv)
- $J$ (warehouses): all Warehouse ID in `file_0_view_0` (capacity.csv)
- $v_i$: Value column in `file_1_view_0`, mapped by ProductName
- $w_i$: Weight column in `file_1_view_0`, mapped by ProductName
- $C_j$: Capacity column in `file_0_view_0`, mapped by Warehouse ID
- $x_{ij}$: number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)