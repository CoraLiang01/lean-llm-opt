#### Abstract Mathematical Model

**Sets:**
- $I$: Set of display areas (indexed by $i$), with business identifier DisplayID from file_0_view_0.
- $J$: Set of vessel types (indexed by $j$), with business identifier ProductName from file_1_view_0.

**Parameters:**
- $c_i$: Capacity of display area $i$ (from Capacity column in file_0_view_0, indexed by DisplayID).
- $v_j$: Value of vessel type $j$ (from Value column in file_1_view_0, indexed by ProductName).
- $w_j$: Size (Weight) of vessel type $j$ (from Weight column in file_1_view_0, indexed by ProductName).

**Decision Variables:**
- $x_{ij}$: Number of vessels of type $j$ to be placed in display area $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
- Display area capacity constraints:
  \[
  \sum_{j \in J} w_j \cdot x_{ij} \leq c_i, \quad \forall i \in I
  \]
- Nonnegativity and integrality:
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
  \]

---

#### Data Mapping

- $I$ (Display areas): file_0_view_0, column DisplayID
- $c_i$: file_0_view_0, column Capacity, indexed by DisplayID
- $J$ (Vessel types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

All parameters and sets are to be used exactly as returned, preserving original file and column names for mapping.