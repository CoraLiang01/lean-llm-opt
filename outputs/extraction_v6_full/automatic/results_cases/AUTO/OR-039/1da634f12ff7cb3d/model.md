## Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (indexed by $i$), from `file_1_view_0.ProductName`
- $J$: Set of warehouses (indexed by $j$), from `file_0_view_0.Warehouse ID`

**Parameters:**
- $b_i$: Value (benefit coefficient) of vehicle type $i$, from `file_1_view_0.Value`
- $w_i$: Weight (space requirement) of vehicle type $i$, from `file_1_view_0.Weight`
- $C_j$: Capacity of warehouse $j$, from `file_0_view_0.Capacity$

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} b_i \cdot x_{ij}
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

## Data Mapping

- $I$ (vehicle types): All `ProductName` in `file_1_view_0`
- $J$ (warehouses): All `Warehouse ID` in `file_0_view_0`
- $b_i$: `file_1_view_0.Value` (indexed by `ProductName`)
- $w_i$: `file_1_view_0.Weight` (indexed by `ProductName`)
- $C_j$: `file_0_view_0.Capacity` (indexed by `Warehouse ID`)
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (decision variable, integer, nonnegative)

---

**Note:** All index sets, parameters, and mappings are derived directly from the validated source tables and columns as required.