#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of storage areas (indexed by $i$), from `file_0_view_0.StorageID`
- $J$: Set of air conditioner types (indexed by $j$), from `file_1_view_0.ProductName`

**Parameters:**
- $C_i$: Capacity of storage area $i$, from `file_0_view_0.Capacity`
- $v_j$: Value of one unit of air conditioner type $j$, from `file_1_view_0.Value`
- $w_j$: Size (weight) of one unit of air conditioner type $j$, from `file_1_view_0.Weight`

**Decision Variables:**
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Capacity Constraints (for each storage area):**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (storage areas): `file_0_view_0.StorageID` (from "capacity.csv")
- $C_i$ (capacity): `file_0_view_0.Capacity` (from "capacity.csv")
- $J$ (air conditioner types): `file_1_view_0.ProductName` (from "products.csv")
- $v_j$ (value): `file_1_view_0.Value` (from "products.csv")
- $w_j$ (weight/size): `file_1_view_0.Weight` (from "products.csv")
- $x_{ij}$: Number of units of product $j$ in storage area $i$ (decision variable, not in source)

No additional constraints or relationships are imposed by the query or data. All index sets and parameters are defined by the full data in the referenced columns.