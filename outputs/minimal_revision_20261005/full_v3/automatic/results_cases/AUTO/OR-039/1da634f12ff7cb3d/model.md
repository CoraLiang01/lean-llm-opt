**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from `file_1_view_0`, column `ProductName`)
- $J$: Set of warehouses, indexed by $j$ (from `file_0_view_0`, column `Warehouse ID`)

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from `file_1_view_0`, column `Value`)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from `file_1_view_0`, column `Weight`)
- $C_j$: Capacity of warehouse $j$ (from `file_0_view_0`, column `Capacity`)

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

**Constraints:**
1. **Warehouse Capacity Constraints:**  
   For each warehouse $j \in J$,
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j
   \]
2. **Nonnegativity and Integrality:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $I$ (vehicle types): `file_1_view_0`, column `ProductName`
- $J$ (warehouses): `file_0_view_0`, column `Warehouse ID`
- $v_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C_j$: `file_0_view_0`, column `Capacity`, keyed by `Warehouse ID`