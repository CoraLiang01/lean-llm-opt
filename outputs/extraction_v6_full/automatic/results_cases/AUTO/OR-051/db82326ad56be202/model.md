#### Abstract Mathematical Model

**Sets:**
- $I$: set of cabinets, indexed by $i$ (CabinetID from file_0_view_0)
- $J$: set of coffee products, indexed by $j$ (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: capacity of cabinet $i$ (capacity, file_0_view_0, CabinetID)
- $v_j$: value per unit of product $j$ (value, file_1_view_0, ProductName)
- $w_j$: weight per unit of product $j$ (weight, file_1_view_0, ProductName)

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
- Cabinet capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i
\]
- Integrality and nonnegativity (for all $i \in I$, $j \in J$):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $c_i$: file_0_view_0, column CabinetID (business key), value column Capacity
- $v_j$: file_1_view_0, column ProductName (business key), value column Value
- $w_j$: file_1_view_0, column ProductName (business key), value column Weight

Each $x_{ij}$ is indexed by CabinetID and ProductName as provided in the source files.