#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of platforms (from capacity.csv, column PlatformId)
- $J$: set of game genres (from products.csv, column ProductName)

**Parameters:**
- $C_i$: memory capacity of platform $i \in I$ (from capacity.csv, column Capacity)
- $v_j$: value per unit of genre $j \in J$ (from products.csv, column Value)
- $w_j$: memory requirement per unit of genre $j \in J$ (from products.csv, column Weight)

**Decision Variables:**
- $x_{ij}$: integer number of units of genre $j$ to be listed on platform $i$, $\forall i \in I, j \in J$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**

1. **Platform Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Variable Domain Constraints:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, j \in J
   \]

---

#### Data Mapping

- Table: capacity.csv, table_id: file_0_view_0
  - Platform index set $I$: column PlatformId
  - Platform capacity $C_i$: column Capacity

- Table: products.csv, table_id: file_1_view_0
  - Genre index set $J$: column ProductName
  - Value per unit $v_j$: column Value
  - Memory requirement per unit $w_j$: column Weight

No additional filters were applied; the full datasets are used as returned by CSVQA.