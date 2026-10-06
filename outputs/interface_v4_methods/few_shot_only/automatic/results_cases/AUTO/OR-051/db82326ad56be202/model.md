#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of cabinets (indexed by $i$), from `capacity.csv` column `CabinetID`
- $J$: set of coffee products (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: capacity of cabinet $i$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value per unit of product $j$ (from `products.csv`, column `Value`)
- $w_j$: weight per unit of product $j$ (from `products.csv`, column `Weight`)

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Cabinet Capacity Constraints:**  
   For each cabinet $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: `capacity.csv`, column `CabinetID`, table_id: `file_0_view_0`
- $C_i$: `capacity.csv`, column `Capacity`, table_id: `file_0_view_0`
- $J$: `products.csv`, column `ProductName`, table_id: `file_1_view_0`
- $v_j$: `products.csv`, column `Value`, table_id: `file_1_view_0`
- $w_j$: `products.csv`, column `Weight`, table_id: `file_1_view_0`
- $x_{ij}$: integer variable for each $(i,j)$

All cabinets and products from the source files are included, preserving their original order and identifiers.