#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (indexed by $i$), from file_1_view_0[ProductName]
- $J$: Set of warehouses (indexed by $j$), from file_0_view_0[Warehouse ID]

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (file_1_view_0[Value])
- $w_i$: Weight (space requirement) of vehicle type $i$ (file_1_view_0[Weight])
- $C_j$: Capacity of warehouse $j$ (file_0_view_0[Capacity])

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

**Constraints:**
1. **Warehouse Capacity Constraints:**
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j, \quad \forall j \in J
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (vehicle types): file_1_view_0[ProductName]
- $J$ (warehouses): file_0_view_0[Warehouse ID]
- $v_i$: file_1_view_0[Value], keyed by ProductName
- $w_i$: file_1_view_0[Weight], keyed by ProductName
- $C_j$: file_0_view_0[Capacity], keyed by Warehouse ID

All parameters and index sets are to be taken directly from the corresponding columns and rows of the retrieved CSV files, preserving their original order and identifiers.