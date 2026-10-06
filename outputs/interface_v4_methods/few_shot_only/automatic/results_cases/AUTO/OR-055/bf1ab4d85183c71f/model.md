#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of display areas (indexed by $i$), with business identifier DisplayID from file_0_view_0.
- $J$: Set of boat types (indexed by $j$), with business identifier ProductName from file_1_view_0.

**Parameters:**
- $c_i$: Capacity of display area $i$ (from Capacity column in file_0_view_0).
- $v_j$: Value of one unit of boat type $j$ (from Value column in file_1_view_0).
- $w_j$: Size (Weight) of one unit of boat type $j$ (from Weight column in file_1_view_0).

**Decision Variables:**
- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
- **Display Area Capacity Constraints:**  
  For each display area $i \in I$,
  \[
  \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
  \]

- **Integrality and Nonnegativity:**  
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
  \]

---

#### Data Mapping

- $I$ (Display areas): file_0_view_0, column DisplayID
- $c_i$: file_0_view_0, column Capacity, keyed by DisplayID
- $J$ (Boat types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

All parameters and index sets are mapped directly from the original files and columns as returned by CSVQA. No data or columns are omitted.